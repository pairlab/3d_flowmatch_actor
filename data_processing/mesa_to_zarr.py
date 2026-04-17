import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import zarr
from numcodecs import Blosc
from scipy.spatial.transform import Rotation


CAMERA_ORDER = ("egocentric", "robot0_eye_in_hand", "robot1_eye_in_hand")
NUM_CAMERAS = 3
NUM_HANDS = 2
NUM_HISTORY = 3
DOF_PER_HAND = 8
ZNEAR = 0.001
ZFAR = 50.0
EXTENT = 11.831  # Mesa MJCF scene characteristic length; required for metric depth
JAW_MIN = 0.079
JAW_MAX = 0.121
DEFAULT_MOTION_THRESHOLD = 0.02


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert mesa-bimanual MimicGen HDF5 trajectories to 3DFA Zarr."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_hdf5",
        type=str,
        help="Path to a single HDF5 demo file (single-task mode).",
    )
    group.add_argument(
        "--input_dir",
        type=str,
        help=(
            "Path to directory containing task subdirectories, each with "
            "demo/demo.hdf5 (multi-task mode)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory. Writes {output_dir}/{mode}/train.zarr and val.zarr.",
    )
    parser.add_argument(
        "--mode",
        choices=("keypose", "dense", "both"),
        default="both",
        help="Sampling mode(s) to generate.",
    )
    parser.add_argument(
        "--motion_threshold",
        type=float,
        default=DEFAULT_MOTION_THRESHOLD,
        help="Motion threshold (meters) for the legacy MESA keypose heuristic.",
    )
    parser.add_argument(
        "--keypose_method",
        choices=("mesa", "peract2"),
        default="mesa",
        help=(
            "Keypose heuristic to use. 'mesa' keeps the original motion-threshold "
            "logic; 'peract2' mirrors the RLBench/PerAct2 stop-and-gripper-change heuristic."
        ),
    )
    parser.add_argument(
        "--stopping_delta",
        type=float,
        default=0.1,
        help="Joint-velocity tolerance for the PerAct2-style keypose heuristic.",
    )
    parser.add_argument(
        "--action_horizon",
        type=int,
        default=1,
        help="Number of future actions to store per sample.",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default=None,
        help=(
            "Path to the training instructions JSON.  When provided, task_ids "
            "are assigned based on each task's position in the JSON key order, "
            "ensuring alignment with the training dataset for subset builds."
        ),
    )
    return parser.parse_args()


def _cast_float32(array):
    if np.issubdtype(array.dtype, np.floating) and array.dtype != np.float32:
        return array.astype(np.float32)
    return array


def _normalize_rows(array, eps=1e-8):
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.clip(norms, eps, None)


def rot6d_to_quat_xyzw(rot6d):
    """
    Convert rot6d representation (first two matrix columns) to quaternion xyzw.
    rot6d: (T, 6) float32
    returns: (T, 4) float32
    """
    rot6d = _cast_float32(rot6d)
    c1 = _normalize_rows(rot6d[:, :3])
    c2 = rot6d[:, 3:6] - np.sum(rot6d[:, 3:6] * c1, axis=-1, keepdims=True) * c1

    c2_norm = np.linalg.norm(c2, axis=-1)
    needs_fallback = c2_norm < 1e-6
    if np.any(needs_fallback):
        fallback_axes = np.zeros_like(c1)
        use_x_axis = np.abs(c1[:, 0]) < 0.9
        fallback_axes[use_x_axis] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        fallback_axes[~use_x_axis] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        c2_fallback = np.cross(c1, fallback_axes)
        c2[needs_fallback] = c2_fallback[needs_fallback]

    c2 = _normalize_rows(c2)
    c3 = _normalize_rows(np.cross(c1, c2))
    rot_mats = np.stack((c1, c2, c3), axis=-1)
    quats = Rotation.from_matrix(rot_mats).as_quat()
    return quats.astype(np.float32)


def action_ee_pose_to_8dof(abs_actions_ee_pose):
    """
    Convert (T, 20) MimicGen abs ee pose action to (T, 2, 8):
    [x, y, z, qx, qy, qz, qw, grip] for each hand.
    """
    abs_actions_ee_pose = _cast_float32(abs_actions_ee_pose)
    t_steps = abs_actions_ee_pose.shape[0]
    output = np.empty((t_steps, NUM_HANDS, DOF_PER_HAND), dtype=np.float32)

    hand_specs = (
        (0, 3, 9),    # robot0
        (10, 13, 19),  # robot1
    )
    for hand_idx, (pos_start, rot_start, grip_idx) in enumerate(hand_specs):
        output[:, hand_idx, :3] = abs_actions_ee_pose[:, pos_start:pos_start + 3]
        output[:, hand_idx, 3:7] = rot6d_to_quat_xyzw(
            abs_actions_ee_pose[:, rot_start:rot_start + 6]
        )
        output[:, hand_idx, 7] = np.clip(
            (abs_actions_ee_pose[:, grip_idx] + 1.0) / 2.0, 0.0, 1.0
        )
    return output


def build_proprioception(obs_group):
    """
    Build proprio states of shape (T, 2, 8):
    [eef_pos(3), eef_quat(4), gripper_norm(1)] per hand.
    """
    hands = []
    for hand_idx in range(NUM_HANDS):
        eef_pos = _cast_float32(obs_group[f"robot{hand_idx}_eef_pos"][:])
        eef_quat = _cast_float32(obs_group[f"robot{hand_idx}_eef_quat"][:])
        jaw_width = _cast_float32(obs_group[f"robot{hand_idx}_gripper_jaw_width"][:])
        grip_norm = np.clip((jaw_width - JAW_MIN) / (JAW_MAX - JAW_MIN), 0.0, 1.0)
        hand_state = np.concatenate(
            (eef_pos, eef_quat, grip_norm[:, None].astype(np.float32)),
            axis=-1,
        ).astype(np.float32)
        hands.append(hand_state)
    return np.stack(hands, axis=1).astype(np.float32)


def build_history(states, timesteps, num_history=NUM_HISTORY):
    """
    states: (T, ...)
    timesteps: (N,)
    returns: (N, num_history, ...)
    """
    timesteps = np.asarray(timesteps, dtype=np.int64)
    offsets = np.arange(-num_history + 1, 1, dtype=np.int64)
    src_idx = np.clip(timesteps[:, None] + offsets[None, :], 0, states.shape[0] - 1)
    return states[src_idx]


def linearize_depth(z_buffer):
    z_buffer = _cast_float32(z_buffer)
    return ZNEAR * EXTENT / (1.0 - z_buffer * (1.0 - ZNEAR / ZFAR))


def extract_observation_tensors(obs_group):
    rgb = []
    depth = []
    extrinsics = []
    intrinsics = []

    for cam in CAMERA_ORDER:
        rgb_cam = obs_group[f"{cam}_image"][:]  # (T, H, W, 3)
        rgb_cam = rgb_cam.transpose(0, 3, 1, 2)  # (T, 3, H, W)
        rgb.append(rgb_cam.astype(np.uint8))

        z_buffer = _cast_float32(obs_group[f"{cam}_depth"][:]).squeeze(-1)
        depth_cam = linearize_depth(z_buffer).astype(np.float16)
        depth.append(depth_cam)

        extrinsics.append(_cast_float32(obs_group[f"{cam}_extrinsic"][:]).astype(np.float16))
        intrinsics.append(_cast_float32(obs_group[f"{cam}_intrinsic"][:]).astype(np.float16))

    return (
        np.stack(rgb, axis=1),        # (T, 3, 3, H, W)
        np.stack(depth, axis=1),      # (T, 3, H, W)
        np.stack(extrinsics, axis=1),  # (T, 3, 4, 4)
        np.stack(intrinsics, axis=1),  # (T, 3, 3, 3)
    )


def detect_keyposes(action_states, motion_threshold):
    """
    action_states: (T, 2, 8)
    heuristic:
    - first and last timestep
    - gripper state change in either hand
    - position delta above threshold in either hand
    """
    num_steps = action_states.shape[0]
    if num_steps == 0:
        return np.array([], dtype=np.int64)
    if num_steps == 1:
        return np.array([0], dtype=np.int64)

    grip = action_states[:, :, 7]
    grip_change = np.any(np.abs(np.diff(grip, axis=0)) > 1e-6, axis=1)

    pos = action_states[:, :, :3]
    pos_delta = np.linalg.norm(pos[1:] - pos[:-1], axis=-1)
    large_motion = np.any(pos_delta > motion_threshold, axis=1)

    key_idxs = np.where(grip_change | large_motion)[0] + 1
    key_idxs = np.unique(np.concatenate(([0], key_idxs, [num_steps - 1])))
    return key_idxs.astype(np.int64)


def _gripper_open_from_obs(obs_group):
    grip_open = []
    for hand_idx in range(NUM_HANDS):
        jaw_width = _cast_float32(obs_group[f"robot{hand_idx}_gripper_jaw_width"][:])
        grip_norm = np.clip((jaw_width - JAW_MIN) / (JAW_MAX - JAW_MIN), 0.0, 1.0)
        grip_open.append(grip_norm > 0.5)
    return np.stack(grip_open, axis=1)


def _joint_vel_from_obs(obs_group):
    joint_vel = []
    for hand_idx in range(NUM_HANDS):
        joint_vel.append(_cast_float32(obs_group[f"robot{hand_idx}_joint_vel"][:]))
    return np.stack(joint_vel, axis=1)


def _is_stopped_peract2(grip_open, joint_vel, i, hand_idx, stopping_delta):
    num_steps = grip_open.shape[0]
    next_is_not_final = i == (num_steps - 2)
    gripper_state_no_change = i < (num_steps - 2) and (
        grip_open[i, hand_idx] == grip_open[i + 1, hand_idx]
        and grip_open[i, hand_idx] == grip_open[max(0, i - 1), hand_idx]
        and grip_open[max(0, i - 2), hand_idx] == grip_open[max(0, i - 1), hand_idx]
    )
    small_delta = np.allclose(joint_vel[i, hand_idx], 0, atol=stopping_delta)
    return small_delta and (not next_is_not_final) and gripper_state_no_change


def detect_keyposes_peract2(obs_group, stopping_delta):
    """
    Mirror the RLBench/PerAct2 heuristic:
    - keypose on either gripper state change
    - keypose when both arms are stopped
    - always include the last timestep
    The initial frame is added separately by keypose_indices().
    """
    grip_open = _gripper_open_from_obs(obs_group)
    joint_vel = _joint_vel_from_obs(obs_group)
    num_steps = grip_open.shape[0]
    if num_steps <= 1:
        return np.array([], dtype=np.int64)

    episode_keypoints = []
    prev_grip_open = grip_open[0].copy()
    stopped_buffer = 0

    for i in range(num_steps):
        right_stopped = _is_stopped_peract2(grip_open, joint_vel, i, 0, stopping_delta)
        left_stopped = _is_stopped_peract2(grip_open, joint_vel, i, 1, stopping_delta)
        stopped = (stopped_buffer <= 0) and right_stopped and left_stopped
        stopped_buffer = 4 if stopped else stopped_buffer - 1

        last = i == (num_steps - 1)
        state_changed = np.any(grip_open[i] != prev_grip_open)
        if i != 0 and (state_changed or last or stopped):
            episode_keypoints.append(i)

        prev_grip_open = grip_open[i].copy()

    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)

    return np.asarray(episode_keypoints, dtype=np.int64)


def dense_indices(num_steps, action_horizon=1):
    current = np.arange(num_steps, dtype=np.int64)
    offsets = np.arange(1, action_horizon + 1, dtype=np.int64)
    target = np.minimum(current[:, None] + offsets[None, :], num_steps - 1)
    return current, target


def keypose_indices(
    action_states,
    obs_group,
    keypose_method,
    motion_threshold,
    stopping_delta,
    action_horizon=1,
):
    if keypose_method == "peract2":
        key_idx = detect_keyposes_peract2(obs_group, stopping_delta)
        key_idx = np.concatenate(([0], key_idx))
    else:
        key_idx = detect_keyposes(action_states, motion_threshold)

    if key_idx.size <= 1:
        return np.array([], dtype=np.int64), np.empty((0, action_horizon), dtype=np.int64)

    src_idx = key_idx[:-1]
    offsets = np.arange(1, action_horizon + 1, dtype=np.int64)
    target_pos = np.minimum(
        np.arange(src_idx.size, dtype=np.int64)[:, None] + offsets[None, :],
        key_idx.size - 1,
    )
    target = key_idx[target_pos]
    return src_idx, target


def create_zarr(path, action_horizon=1):
    compressor = Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE)
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    group = zarr.open_group(str(path), mode="w")

    def _create(name, shape, dtype):
        group.create_dataset(
            name,
            shape=(0,) + shape,
            chunks=(1,) + shape,
            compressor=compressor,
            dtype=dtype,
        )

    _create("rgb", (NUM_CAMERAS, 3, 128, 128), "uint8")
    _create("depth", (NUM_CAMERAS, 128, 128), "float16")
    _create("proprioception", (NUM_HISTORY, NUM_HANDS, DOF_PER_HAND), "float32")
    _create("action", (action_horizon, NUM_HANDS, DOF_PER_HAND), "float32")
    _create("extrinsics", (NUM_CAMERAS, 4, 4), "float16")
    _create("intrinsics", (NUM_CAMERAS, 3, 3), "float16")
    _create("task_id", (), "uint8")
    _create("variation", (), "uint8")
    return group


def sorted_demo_keys(data_group):
    def _demo_index(name):
        if "_" in name:
            suffix = name.rsplit("_", maxsplit=1)[-1]
            if suffix.isdigit():
                return int(suffix)
        return name

    return sorted(data_group.keys(), key=_demo_index)


def append_demo_samples(
    zarr_group,
    demo_group,
    mode,
    keypose_method,
    motion_threshold,
    stopping_delta,
    action_horizon=1,
    task_id=0,
):
    obs = demo_group["obs"]
    action_states = action_ee_pose_to_8dof(demo_group["abs_actions_ee_pose"][:])
    proprio_states = build_proprioception(obs)
    rgb, depth, extrinsics, intrinsics = extract_observation_tensors(obs)

    if mode == "dense":
        src_idx, tgt_idx = dense_indices(
            action_states.shape[0], action_horizon=action_horizon
        )
    else:
        src_idx, tgt_idx = keypose_indices(
            action_states,
            obs,
            keypose_method,
            motion_threshold,
            stopping_delta,
            action_horizon=action_horizon,
        )

    proprio_hist = build_history(proprio_states, src_idx).astype(np.float32)
    action = action_states[tgt_idx].astype(np.float32)

    n = int(src_idx.shape[0])
    zarr_group["rgb"].append(rgb[src_idx].astype(np.uint8))
    zarr_group["depth"].append(depth[src_idx].astype(np.float16))
    zarr_group["proprioception"].append(proprio_hist)
    zarr_group["action"].append(action)
    zarr_group["extrinsics"].append(extrinsics[src_idx].astype(np.float16))
    zarr_group["intrinsics"].append(intrinsics[src_idx].astype(np.float16))
    zarr_group["task_id"].append(np.full((n,), task_id, dtype=np.uint8))
    zarr_group["variation"].append(np.zeros((n,), dtype=np.uint8))

    return n, int(action_states.shape[0])


def convert_single_task(
    task_name,
    task_id,
    hdf5_path,
    output_path,
    mode,
    keypose_method,
    motion_threshold,
    stopping_delta,
    action_horizon,
):
    """Convert one task's HDF5 to a standalone Zarr. Process-safe."""
    zarr_group = create_zarr(output_path, action_horizon=action_horizon)
    task_samples = 0
    task_steps = 0
    with h5py.File(hdf5_path, "r") as f:
        data_group = f["data"]
        demo_keys = sorted_demo_keys(data_group)
        num_demos = len(demo_keys)
        for i, demo_key in enumerate(demo_keys):
            num_samples, num_steps = append_demo_samples(
                zarr_group=zarr_group,
                demo_group=data_group[demo_key],
                mode=mode,
                keypose_method=keypose_method,
                motion_threshold=motion_threshold,
                stopping_delta=stopping_delta,
                action_horizon=action_horizon,
                task_id=task_id,
            )
            task_samples += num_samples
            task_steps += num_steps
            print(f"  [{task_name}] demo {i+1}/{num_demos}: {num_samples} samples ({num_steps} steps)", flush=True)
    return task_name, task_id, task_samples, task_steps


def _merge_zarrs(tmp_dir, task_entries, final_path, action_horizon=1, chunk_size=256):
    """Concatenate per-task Zarrs into a single final Zarr in task order."""
    final = create_zarr(final_path, action_horizon=action_horizon)
    for task_name, _, _ in task_entries:
        src = zarr.open_group(str(tmp_dir / f"{task_name}.zarr"), mode="r")
        n = src[list(src.keys())[0]].shape[0]
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            for key in final.keys():
                final[key].append(src[key][start:end])
        print(f"  Merged {task_name}: {n} samples", flush=True)
    return final


def build_mode_dataset(
    task_entries,
    output_dir,
    mode,
    keypose_method,
    motion_threshold,
    stopping_delta,
    action_horizon,
):
    """
    task_entries: list of (task_name, task_id, hdf5_path)
    """
    train_zarr = Path(output_dir) / mode / "train.zarr"
    val_zarr = Path(output_dir) / mode / "val.zarr"
    tmp_dir = Path(output_dir) / mode / "_tmp"

    print(f"\nBuilding mode='{mode}'", flush=True)

    num_tasks = len(task_entries)
    max_workers = min(num_tasks, os.cpu_count() or 1)

    # Phase 1: parallel per-task conversion
    tmp_dir.mkdir(parents=True, exist_ok=True)
    total_samples = 0
    total_steps = 0

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for task_name, task_id, hdf5_path in task_entries:
            tmp_path = str(tmp_dir / f"{task_name}.zarr")
            fut = pool.submit(
                convert_single_task,
                task_name,
                task_id,
                hdf5_path,
                tmp_path,
                mode,
                keypose_method,
                motion_threshold,
                stopping_delta,
                action_horizon,
            )
            futures[fut] = task_name

        for fut in as_completed(futures):
            task_name, task_id, task_samples, task_steps = fut.result()
            total_samples += task_samples
            total_steps += task_steps
            print(f"  {task_name} (id={task_id}): {task_samples} samples from {task_steps} steps", flush=True)

    # Phase 2: merge per-task Zarrs into final Zarr (deterministic task order)
    print("  Merging per-task Zarrs...", flush=True)
    _merge_zarrs(
        tmp_dir,
        task_entries,
        train_zarr,
        action_horizon=action_horizon,
    )

    # Cleanup temp
    shutil.rmtree(tmp_dir)

    # Copy train → val (overfitting setup)
    if val_zarr.exists():
        shutil.rmtree(val_zarr)
    shutil.copytree(train_zarr, val_zarr)

    print(f"Finished mode='{mode}': {total_samples} samples from {total_steps} raw steps", flush=True)
    print(f"  train: {train_zarr}", flush=True)
    print(f"  val:   {val_zarr}", flush=True)


def discover_tasks(input_dir, instructions_file=None):
    """
    Discover task subdirectories and return sorted (task_name, task_id, hdf5_path).

    When *instructions_file* is provided, task_ids are assigned based on each
    task's position in the JSON key order — matching the training dataset's
    task list even for subset builds.  Without it, task_ids are assigned by
    local alphabetical enumeration (correct only when all tasks are present).
    """
    input_dir = Path(input_dir)
    task_names = sorted(
        d.name
        for d in input_dir.iterdir()
        if d.is_dir() and (d / "demo" / "demo.hdf5").exists()
    )
    if instructions_file is not None:
        ref_tasks = list(json.load(open(instructions_file)).keys())
        ref_index = {name: idx for idx, name in enumerate(ref_tasks)}
    entries = []
    for i, task_name in enumerate(task_names):
        if instructions_file is not None:
            if task_name not in ref_index:
                raise ValueError(
                    f"Task '{task_name}' not in instructions file "
                    f"'{instructions_file}'"
                )
            task_id = ref_index[task_name]
        else:
            task_id = i
        hdf5_path = str(input_dir / task_name / "demo" / "demo.hdf5")
        entries.append((task_name, task_id, hdf5_path))
    return entries


def main():
    args = parse_args()
    if args.action_horizon < 1:
        raise ValueError("--action_horizon must be >= 1")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.input_dir:
        task_entries = discover_tasks(args.input_dir, args.instructions)
        print(f"Discovered {len(task_entries)} tasks in {args.input_dir}")
    else:
        task_entries = [("single_task", 0, args.input_hdf5)]

    modes = ("keypose", "dense") if args.mode == "both" else (args.mode,)
    for mode in modes:
        build_mode_dataset(
            task_entries=task_entries,
            output_dir=args.output_dir,
            mode=mode,
            keypose_method=args.keypose_method,
            motion_threshold=args.motion_threshold,
            stopping_delta=args.stopping_delta,
            action_horizon=args.action_horizon,
        )


if __name__ == "__main__":
    main()
