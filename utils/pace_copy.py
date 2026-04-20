"""Copy Zarr data to node-local storage for faster I/O."""

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch.distributed as dist


_STAGING_MARKER = ".staging_complete"
_DEFAULT_WORKERS = 32
_VERIFY_MAX_ATTEMPTS = 3


def _copy_file(src_file: Path, dest_file: Path) -> None:
    shutil.copyfile(src_file, dest_file)


def _enumerate_files(root: Path) -> set[Path]:
    """Return all files under ``root`` as paths relative to ``root``."""
    relative_files: set[Path] = set()
    for dirpath, _, filenames in os.walk(root):
        rel_root = Path(dirpath).relative_to(root)
        for name in filenames:
            relative_files.add(rel_root / name)
    return relative_files


def _copy_files(src: Path, dest: Path, rels, num_workers: int) -> None:
    def _task(rel: Path) -> None:
        (dest / rel.parent).mkdir(parents=True, exist_ok=True)
        _copy_file(src / rel, dest / rel)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for _ in pool.map(_task, rels):
            pass


def _parallel_copytree(src: Path, dest: Path, num_workers: int) -> set[Path]:
    """Copy a directory tree in parallel using a thread pool.

    Zarr stores contain one small chunk file per sample per array, so a
    single-threaded ``shutil.copytree`` bottlenecks on per-file NFS metadata
    round-trips rather than bandwidth. ``shutil.copyfile`` releases the GIL
    during the underlying ``read``/``write`` syscalls, so a thread pool
    delivers near-linear speedup up to the NFS server's concurrency limit
    without the overhead of process fork-and-pickle.
    """
    relative_files = _enumerate_files(src)
    _copy_files(src, dest, relative_files, num_workers)
    return relative_files


def _verify_and_heal(
    src: Path,
    dest: Path,
    num_workers: int,
    max_attempts: int = _VERIFY_MAX_ATTEMPTS,
) -> None:
    """Re-walk src and dest, copy any missing files, retry up to max_attempts.

    The parallel copy has silently dropped individual chunk files under
    concurrent NFS load — evidenced by downstream ``KeyError`` on specific
    zarr chunks thousands of training steps later (see e.g. job 6651861
    crashing on ``val.zarr/action/6548.0.0.0``). Rather than trust the
    first pass, re-enumerate both trees and heal the diff before writing
    the completion marker so training never starts on a partial stage.
    """
    for attempt in range(1, max_attempts + 1):
        src_files = _enumerate_files(src)
        dest_files = _enumerate_files(dest) - {Path(_STAGING_MARKER)}
        missing = src_files - dest_files
        if not missing:
            return
        print(
            f"[pace_copy] verify attempt {attempt}/{max_attempts}: "
            f"{len(missing)} of {len(src_files)} files missing at {dest}; "
            f"re-copying",
            flush=True,
        )
        _copy_files(src, dest, missing, num_workers)

    raise RuntimeError(
        f"pace_copy: staging to {dest} still incomplete after "
        f"{max_attempts} heal attempts ({len(missing)} files missing)"
    )


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
            _verify_and_heal(src, dest, num_workers=num_workers)
            marker.touch()
            print(f"Staged to {dest}", flush=True)

    dist.barrier()
    return dest
