"""Copy Zarr data to node-local storage for faster I/O."""

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch.distributed as dist


_STAGING_MARKER = ".staging_complete"
_DEFAULT_WORKERS = 32


def _copy_file(src_file: Path, dest_file: Path) -> None:
    shutil.copyfile(src_file, dest_file)


def _parallel_copytree(src: Path, dest: Path, num_workers: int) -> None:
    """Copy a directory tree in parallel using a thread pool.

    Zarr stores contain one small chunk file per sample per array, so a
    single-threaded ``shutil.copytree`` bottlenecks on per-file NFS metadata
    round-trips rather than bandwidth. ``shutil.copyfile`` releases the GIL
    during the underlying ``read``/``write`` syscalls, so a thread pool
    delivers near-linear speedup up to the NFS server's concurrency limit
    without the overhead of process fork-and-pickle.
    """
    # Phase 1: walk the source tree once, creating all destination
    # directories up front so per-file workers never race on mkdir.
    relative_files: list[Path] = []
    for root, _, filenames in os.walk(src):
        rel_root = Path(root).relative_to(src)
        (dest / rel_root).mkdir(parents=True, exist_ok=True)
        for name in filenames:
            relative_files.append(rel_root / name)

    # Phase 2: copy files concurrently. ``executor.map`` propagates the
    # first worker exception on iteration, so a partial copy surfaces as
    # a hard failure rather than silently leaving a truncated tree behind.
    def _task(rel: Path) -> None:
        _copy_file(src / rel, dest / rel)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for _ in pool.map(_task, relative_files):
            pass


def copy_zarr_to_local(src_path, pace_tmp_dir=None, num_workers: int = _DEFAULT_WORKERS):
    """Copy a Zarr directory to node-local storage. Returns the local path.

    Only rank 0 copies; other ranks wait at a barrier. A ``.staging_complete``
    sentinel file inside ``dest`` distinguishes a finished copy from a
    partially-copied tree left behind by a prior crash or preemption:
    partial trees are removed and re-staged, while completed trees are
    reused as-is (idempotent).
    """
    src = Path(src_path)
    local_base = Path(pace_tmp_dir or os.environ["TMPDIR"]) / "3dfa_data"
    dest = local_base / src.name
    marker = dest / _STAGING_MARKER

    if dist.get_rank() == 0:
        if marker.exists():
            print(f"Already staged: {dest}", flush=True)
        else:
            if dest.exists():
                print(f"Removing partial staging at {dest}", flush=True)
                shutil.rmtree(dest)
            local_base.mkdir(parents=True, exist_ok=True)
            print(
                f"Staging {src} -> {dest} "
                f"(parallel copytree, {num_workers} workers)",
                flush=True,
            )
            _parallel_copytree(src, dest, num_workers=num_workers)
            marker.touch()
            print(f"Staged to {dest}", flush=True)

    dist.barrier()
    return dest
