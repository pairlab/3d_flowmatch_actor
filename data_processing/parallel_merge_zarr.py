"""
Parallel merge of per-task Zarr stores into a single contiguous Zarr.

Why this exists: the in-process append-based merge in mesa_to_zarr.py is
serial and bottlenecked on a slow networked filesystem. Per-task Zarrs use
chunks of size 1 along the leading axis, so each sample is an independent
chunk file. That makes it safe to pre-size the destination once and have
many worker processes write disjoint slices in parallel.
"""

import argparse
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import zarr

# Allow `from data_processing.mesa_to_zarr import ...` when invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_processing.mesa_to_zarr import create_zarr, discover_tasks  # noqa: E402


WRITE_CHUNK = 256


def _task_lengths(tmp_dir: Path, task_names):
    lengths = {}
    for name in task_names:
        z = zarr.open_group(str(tmp_dir / f"{name}.zarr"), mode="r")
        lengths[name] = int(z["action"].shape[0])
    return lengths


def _write_task_slice(tmp_dir_str, task_name, dst_offset, dst_path_str, keys):
    """Worker: copy one task's per-key arrays into the destination at dst_offset."""
    src = zarr.open_group(f"{tmp_dir_str}/{task_name}.zarr", mode="r")
    dst = zarr.open_group(dst_path_str, mode="r+")
    n = int(src[keys[0]].shape[0])
    for off in range(0, n, WRITE_CHUNK):
        end = min(off + WRITE_CHUNK, n)
        for key in keys:
            dst[key][dst_offset + off : dst_offset + end] = src[key][off:end]
    return task_name, n


def parallel_merge(tmp_dir: Path, task_entries, dst_path: Path, num_workers: int):
    """
    Pre-size dst_path, then write each task's slice in parallel.

    task_entries must be in the canonical task order (alphabetical, matching
    discover_tasks); offsets are computed in that order so the merged output
    is bit-identical to a serial merge.
    """
    task_names = [name for name, _, _ in task_entries]
    lengths = _task_lengths(tmp_dir, task_names)
    total = sum(lengths.values())
    print(f"Pre-sizing {dst_path} for {total} samples across {len(task_names)} tasks", flush=True)

    dst = create_zarr(dst_path)
    keys = list(dst.keys())
    for key in keys:
        arr = dst[key]
        arr.resize((total,) + arr.shape[1:])

    offsets = {}
    cursor = 0
    for name in task_names:
        offsets[name] = cursor
        cursor += lengths[name]
    assert cursor == total

    print(f"Launching {num_workers} workers for parallel merge", flush=True)
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(
                _write_task_slice,
                str(tmp_dir),
                name,
                offsets[name],
                str(dst_path),
                keys,
            ): name
            for name in task_names
        }
        for fut in as_completed(futures):
            name, n = fut.result()
            print(f"  wrote {name}: {n} samples at offset {offsets[name]}", flush=True)

    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tmp_dir", required=True, help="Directory of per-task zarrs.")
    parser.add_argument("--input_dir", required=True, help="Source HDF5 task dir (for canonical task order).")
    parser.add_argument("--dst_zarr", required=True, help="Destination zarr path (will be created/overwritten).")
    parser.add_argument("--final_train_zarr", default=None, help="Optional: move dst_zarr here after merge.")
    parser.add_argument("--final_val_zarr", default=None, help="Optional: symlink to final_train_zarr after move.")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    tmp_dir = Path(args.tmp_dir)
    dst_path = Path(args.dst_zarr)
    task_entries = discover_tasks(args.input_dir)

    total = parallel_merge(tmp_dir, task_entries, dst_path, args.num_workers)
    print(f"Merge complete: {total} samples -> {dst_path}", flush=True)

    if args.final_train_zarr:
        final_train = Path(args.final_train_zarr)
        if final_train.exists():
            shutil.rmtree(final_train)
        final_train.parent.mkdir(parents=True, exist_ok=True)
        print(f"Moving {dst_path} -> {final_train}", flush=True)
        shutil.move(str(dst_path), str(final_train))
        print("Move complete", flush=True)

        if args.final_val_zarr:
            final_val = Path(args.final_val_zarr)
            if final_val.exists() or final_val.is_symlink():
                if final_val.is_symlink() or final_val.is_file():
                    final_val.unlink()
                else:
                    shutil.rmtree(final_val)
            final_val.symlink_to(final_train.name)
            print(f"Symlinked {final_val} -> {final_train.name}", flush=True)


if __name__ == "__main__":
    main()
