"""Copy Zarr data to node-local storage for faster I/O."""

import os
import shutil
from pathlib import Path

import torch.distributed as dist


def copy_zarr_to_local(src_path, pace_tmp_dir=None):
    """Copy a Zarr directory to node-local storage. Returns the local path.

    Only rank 0 copies; other ranks wait at a barrier.
    Skips if destination already exists (idempotent).
    """
    src = Path(src_path)
    local_base = Path(pace_tmp_dir or os.environ["TMPDIR"]) / "3dfa_data"
    dest = local_base / src.name

    if dist.get_rank() == 0:
        if not dest.exists():
            local_base.mkdir(parents=True, exist_ok=True)
            print(f"Staging {src} -> {dest}")
            shutil.copytree(src, dest)
            print(f"Staged to {dest}")
        else:
            print(f"Already staged: {dest}")

    dist.barrier()
    return dest
