"""Convert mesa-70 HDF5 demos (single-arm, no depth) to 3DFA zarr format.

Layout written per sample:
    rgb           (3, 3, 128, 128)  uint8   — 3 cameras, 3 channels
    depth         (3, 128, 128)     float16 — all zeros (no depth in source)
    proprioception (NUM_HISTORY, 1, 8) float32
    action        (1, 1, 8)         float32 — chunk_size=1 dense target
    extrinsics    (3, 4, 4)         float16 — identity (unused by pixel-grid cloud)
    intrinsics    (3, 3, 3)         float16 — identity (unused by pixel-grid cloud)
    task_id       ()                uint8
    variation     ()                uint8

Usage:
    python data_processing/mesa70_to_zarr.py \\
        --input_dir /path/to/mesa-70-hdf5 \\
        --output_dir /path/to/output \\
        --val_frac 0.2
"""

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import zarr
from numcodecs import Blosc
from scipy.spatial.transform import Rotation


CAMERA_KEYS = ("leftshoulder_image", "rightshoulder_image", "robot0_eye_in_hand_image")
NUM_CAMERAS = 3
NUM_HANDS = 1
NUM_HISTORY = 3
DOF_PER_HAND = 8
IMG_SIZE = 128

JAW_MIN = 0.0
JAW_MAX = 0.121


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert mesa-70 single-arm HDF5 demos to 3DFA zarr."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory with one .hdf5 file per task.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root output dir. Writes {output_dir}/train.zarr and val.zarr.")
    parser.add_argument("--val_frac", type=float, default=0.2,
                        help="Fraction of demos used for validation (per task).")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (0 = sequential).")
    return parser.parse_args()


def _cast_f32(arr):
    return arr.astype(np.float32) if arr.dtype != np.float32 else arr


def action_to_8dof(abs_actions):
    """(T, 7) → (T, 1, 8): pos(3) + aa(3) + grip{-1,+1} → pos+quat+grip[0,1]."""
    abs_actions = _cast_f32(abs_actions)
    T = abs_actions.shape[0]
    out = np.empty((T, 1, DOF_PER_HAND), dtype=np.float32)
    pos = abs_actions[:, :3]
    aa = abs_actions[:, 3:6]
    grip = abs_actions[:, 6]
    quats = Rotation.from_rotvec(aa).as_quat().astype(np.float32)  # xyzw
    grip_norm = np.clip((grip + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
    out[:, 0, :3] = pos
    out[:, 0, 3:7] = quats
    out[:, 0, 7] = grip_norm
    return out


def build_proprio(obs_group):
    """(T, 1, 8): eef_pos + eef_quat_xyzw + grip_norm.

    Source keys actually present in the mesa-70 HDF5:
      hand_mat           (T, 4, 4)  — eef SE(3) transform; [:, :3, 3] = pos, [:, :3, :3] = R
      robot0_eef_pos     (T, 3)     — identical to hand_mat[:, :3, 3], used directly
      robot0_gripper_qpos (T, 2)    — symmetric finger joints; jaw = qpos[:,0] - qpos[:,1]
    JAW_MAX = 0.121 normalises the jaw to [0, 1].
    """
    eef_pos = _cast_f32(obs_group["robot0_eef_pos"][:])       # (T, 3)
    hand_mat = _cast_f32(obs_group["hand_mat"][:, :3, :3])    # (T, 3, 3) rotation
    eef_quat = Rotation.from_matrix(hand_mat).as_quat().astype(np.float32)  # (T, 4) xyzw
    qpos = _cast_f32(obs_group["robot0_gripper_qpos"][:])     # (T, 2)
    jaw = qpos[:, 0] - qpos[:, 1]                             # (T,) symmetric jaw width
    grip_norm = np.clip(jaw / JAW_MAX, 0.0, 1.0)[:, None]     # (T, 1)
    hand = np.concatenate([eef_pos, eef_quat, grip_norm], axis=-1)  # (T, 8)
    return hand[:, None, :]  # (T, 1, 8)


def build_history(states, timesteps, num_history=NUM_HISTORY):
    """states (T, ...) → (N, num_history, ...) with pad-before at boundaries."""
    timesteps = np.asarray(timesteps, dtype=np.int64)
    offsets = np.arange(-num_history + 1, 1, dtype=np.int64)
    src = np.clip(timesteps[:, None] + offsets[None, :], 0, states.shape[0] - 1)
    return states[src]


def resize_images(rgb_hwc, size=IMG_SIZE):
    """(T, H, W, 3) uint8 → (T, 3, size, size) uint8 via simple subsampling."""
    try:
        from PIL import Image
        T = rgb_hwc.shape[0]
        out = np.empty((T, 3, size, size), dtype=np.uint8)
        for t in range(T):
            img = Image.fromarray(rgb_hwc[t]).resize((size, size), Image.BILINEAR)
            out[t] = np.array(img).transpose(2, 0, 1)
        return out
    except ImportError:
        # fallback: nearest-neighbour via stride
        h, w = rgb_hwc.shape[1], rgb_hwc.shape[2]
        row_idx = np.round(np.linspace(0, h - 1, size)).astype(int)
        col_idx = np.round(np.linspace(0, w - 1, size)).astype(int)
        sub = rgb_hwc[:, row_idx][:, :, col_idx]  # (T, size, size, 3)
        return sub.transpose(0, 3, 1, 2)  # (T, 3, size, size)


def create_zarr(path):
    compressor = Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE)
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(path), mode="w")

    def _mk(name, shape, dtype):
        g.create_dataset(name, shape=(0,) + shape,
                         chunks=(1,) + shape, compressor=compressor, dtype=dtype)

    _mk("rgb",           (NUM_CAMERAS, 3, IMG_SIZE, IMG_SIZE), "uint8")
    _mk("depth",         (NUM_CAMERAS, IMG_SIZE, IMG_SIZE),    "float16")
    _mk("proprioception", (NUM_HISTORY, NUM_HANDS, DOF_PER_HAND), "float32")
    _mk("action",        (1, NUM_HANDS, DOF_PER_HAND),         "float32")
    _mk("extrinsics",    (NUM_CAMERAS, 4, 4),                  "float16")
    _mk("intrinsics",    (NUM_CAMERAS, 3, 3),                  "float16")
    _mk("task_id",       (),                                   "uint8")
    _mk("variation",     (),                                   "uint8")
    return g


def sorted_demo_keys(data_group):
    def _idx(name):
        suffix = name.rsplit("_", 1)[-1]
        return int(suffix) if suffix.isdigit() else name
    return sorted(data_group.keys(), key=_idx)


def append_demo(zarr_group, obs_group, action_states, proprio_states, task_id):
    T = action_states.shape[0]
    if T == 0:
        return 0

    # Build rgb stack (T, 3, 3, 128, 128)
    rgb_cams = []
    for key in CAMERA_KEYS:
        rgb_cam = obs_group[key][:]   # (T, H, W, 3)
        rgb_cam = resize_images(rgb_cam)  # (T, 3, 128, 128)
        rgb_cams.append(rgb_cam)
    rgb = np.stack(rgb_cams, axis=1)  # (T, 3, 3, 128, 128)

    # Dense indexing: current → next (or last)
    src_idx = np.arange(T, dtype=np.int64)
    tgt_idx = np.minimum(src_idx + 1, T - 1)

    # Proprioception history
    prop_hist = build_history(proprio_states, src_idx)  # (T, nhist, 1, 8)
    # Action target (single target step, chunk_size=1)
    action = action_states[tgt_idx][:, None, :, :]  # (T, 1, 1, 8)

    # Dummy depth / extrinsics / intrinsics
    depth = np.zeros((T, NUM_CAMERAS, IMG_SIZE, IMG_SIZE), dtype=np.float16)
    extr = np.tile(np.eye(4, dtype=np.float16), (T, NUM_CAMERAS, 1, 1))
    intr = np.tile(np.eye(3, dtype=np.float16), (T, NUM_CAMERAS, 1, 1))

    zarr_group["rgb"].append(rgb[src_idx].astype(np.uint8))
    zarr_group["depth"].append(depth)
    zarr_group["proprioception"].append(prop_hist.astype(np.float32))
    zarr_group["action"].append(action.astype(np.float32))
    zarr_group["extrinsics"].append(extr)
    zarr_group["intrinsics"].append(intr)
    zarr_group["task_id"].append(np.full((T,), task_id, dtype=np.uint8))
    zarr_group["variation"].append(np.zeros((T,), dtype=np.uint8))
    return T


def convert_task(task_name, task_id, hdf5_path, train_path, val_path, val_frac):
    train_grp = create_zarr(train_path)
    val_grp = create_zarr(val_path)

    n_train = n_val = 0
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        demo_keys = sorted_demo_keys(data)
        n_demos = len(demo_keys)
        n_val_demos = max(1, int(n_demos * val_frac))
        train_keys = demo_keys[:-n_val_demos]
        val_keys = demo_keys[-n_val_demos:]

        for keys, grp, label in [(train_keys, train_grp, "train"),
                                  (val_keys, val_grp, "val")]:
            for demo_key in keys:
                demo = data[demo_key]
                obs = demo["obs"]
                action_states = action_to_8dof(demo["abs_actions"][:])
                proprio_states = build_proprio(obs)
                # align lengths (abs_actions may be T+1 vs obs T)
                T = min(action_states.shape[0], proprio_states.shape[0])
                n = append_demo(grp, obs, action_states[:T], proprio_states[:T], task_id)
                if label == "train":
                    n_train += n
                else:
                    n_val += n

    print(f"  [{task_name}] train={n_train} val={n_val} samples", flush=True)
    return task_name, n_train, n_val


def _merge_zarrs(src_paths, dst_path, chunk_size=256):
    dst = create_zarr(dst_path)
    for src_path in src_paths:
        src = zarr.open_group(str(src_path), mode="r")
        n = src["action"].shape[0]
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            for key in dst.keys():
                dst[key].append(src[key][start:end])
    return dst


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from datasets.mesa70 import MESA70_TASKS
    hdf5_files = sorted(input_dir.glob("*.hdf5"))
    task_entries = []
    for f in hdf5_files:
        task_name = f.stem
        if task_name not in MESA70_TASKS:
            print(f"  WARNING: {task_name} not in MESA70_TASKS — skipping", flush=True)
            continue
        task_id = MESA70_TASKS.index(task_name)
        task_entries.append((task_name, task_id, str(f)))
    task_entries.sort(key=lambda x: x[1])
    print(f"Found {len(task_entries)} tasks in {input_dir}")

    tmp_dir = output_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    if args.workers > 0:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for task_name, task_id, hdf5_path in task_entries:
                fut = pool.submit(
                    convert_task,
                    task_name, task_id, hdf5_path,
                    str(tmp_dir / f"{task_name}_train.zarr"),
                    str(tmp_dir / f"{task_name}_val.zarr"),
                    args.val_frac,
                )
                futures[fut] = task_name
            for fut in as_completed(futures):
                fut.result()
    else:
        for task_name, task_id, hdf5_path in task_entries:
            convert_task(
                task_name, task_id, hdf5_path,
                str(tmp_dir / f"{task_name}_train.zarr"),
                str(tmp_dir / f"{task_name}_val.zarr"),
                args.val_frac,
            )

    print("\nMerging train zarrs...", flush=True)
    train_srcs = [tmp_dir / f"{t}_train.zarr" for t, _, _ in task_entries]
    _merge_zarrs(train_srcs, output_dir / "train.zarr")
    print("Merging val zarrs...", flush=True)
    val_srcs = [tmp_dir / f"{t}_val.zarr" for t, _, _ in task_entries]
    _merge_zarrs(val_srcs, output_dir / "val.zarr")

    shutil.rmtree(tmp_dir)
    print(f"\nDone. Output: {output_dir}/train.zarr and val.zarr")


if __name__ == "__main__":
    main()
