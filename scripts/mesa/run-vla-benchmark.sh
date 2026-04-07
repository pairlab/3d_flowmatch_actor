#!/bin/bash
# Thin shim that runs vla-benchmark's eval_server_parallel.py inside its own
# pinned venv. The sibling clone is expected at ../vla-benchmark relative to
# this repo root, and the venv lives outside the repo so it can be shared.
#
# Override `VLA_BENCHMARK_PYTHON` if your venv lives elsewhere.
set -euo pipefail

VLA_BENCHMARK_PYTHON="${VLA_BENCHMARK_PYTHON:-/storage/project/r-agarg35-0/fchang40/venvs/vla-benchmark/bin/python}"
if [ ! -x "$VLA_BENCHMARK_PYTHON" ]; then
    echo "[run-vla-benchmark.sh] python not found at $VLA_BENCHMARK_PYTHON" >&2
    exit 2
fi

cd "$(dirname "$0")/../../../vla-benchmark"
exec "$VLA_BENCHMARK_PYTHON" scripts/eval_server_parallel.py "$@"
