#!/usr/bin/env python3
"""Asyncio supervisor that boots `serve_policy.py` and `eval_server_parallel.py`.

The policy server runs in this repo's venv;
the eval client runs in the sibling vla-benchmark uv venv via the shim
`scripts/mesa/run-vla-benchmark.sh`.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import socket
import sys
from dataclasses import dataclass

RESET = "\033[0m"
COLORS = {
    ("eval", "OUT"): "\033[96m",
    ("eval", "ERR"): "\033[91;1m",
    ("serve", "OUT"): "\033[92m",
    ("serve", "ERR"): "\033[93m",
}

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_VENV = "/storage/project/r-agarg35-0/fchang40/venvs/imitation-venv"
DEFAULT_STATE_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_jaw_width",
    "robot1_eef_pos",
    "robot1_eef_quat",
    "robot1_gripper_jaw_width",
    "egocentric_intrinsic",
    "egocentric_extrinsic",
    "robot0_eye_in_hand_intrinsic",
    "robot0_eye_in_hand_extrinsic",
    "robot1_eye_in_hand_intrinsic",
    "robot1_eye_in_hand_extrinsic",
]
DEFAULT_CAMERA_NAMES = ["egocentric", "robot0_eye_in_hand", "robot1_eye_in_hand"]


async def _pump(stream: asyncio.StreamReader, name: str, stream_type: str) -> None:
    color = COLORS.get((name, stream_type), "")
    while True:
        line = await stream.readline()
        if not line:
            break
        print(
            f"{color}[{name}][{stream_type}] {line.decode(errors='replace').rstrip()}{RESET}",
            flush=True,
        )


@dataclass
class Child:
    name: str
    cmd: list[str]
    proc: asyncio.subprocess.Process | None = None


async def start_child(
    child: Child, cwd: str | None = None, extra_env: dict | None = None
) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)
    child.proc = await asyncio.create_subprocess_exec(
        *child.cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
    )
    assert child.proc.stdout and child.proc.stderr
    asyncio.create_task(_pump(child.proc.stdout, child.name, "OUT"))
    asyncio.create_task(_pump(child.proc.stderr, child.name, "ERR"))


async def terminate_child(child: Child, timeout_s: float = 5.0) -> None:
    if not child.proc or child.proc.returncode is not None:
        return
    child.proc.terminate()
    try:
        await asyncio.wait_for(child.proc.wait(), timeout=timeout_s)
    except asyncio.TimeoutError:
        child.proc.kill()
        await child.proc.wait()


async def wait_port_listening(
    host: str,
    port: int,
    timeout_s: float = 180.0,
    poll_s: float = 0.2,
    proc: asyncio.subprocess.Process | None = None,
) -> None:
    deadline = asyncio.get_event_loop().time() + timeout_s
    while True:
        if proc is not None and proc.returncode is not None:
            raise RuntimeError(f"Server process exited early with code {proc.returncode}")
        try:
            _, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(f"Timed out waiting for {host}:{port} to accept connections")
            await asyncio.sleep(poll_s)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # ---- Supervisor / paths ----
    parser.add_argument("--policy-venv", default=DEFAULT_VENV,
                        help="Virtualenv used by serve_policy.py.")
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=None,
                        help="If unset, picks a free port automatically.")
    parser.add_argument("--server-startup-timeout", type=float, default=300.0)

    # ---- Checkpoint / model shaping (forwarded to serve_policy.py) ----
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pth checkpoint produced by main.py.")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--bimanual", default="True")
    parser.add_argument("--num-history", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=120)
    parser.add_argument("--num-attn-heads", type=int, default=8)
    parser.add_argument("--num-vis-instr-attn-layers", type=int, default=3)
    parser.add_argument("--num-shared-attn-layers", type=int, default=4)
    parser.add_argument("--rotation-format", default="quat_xyzw")
    parser.add_argument("--denoise-timesteps", type=int, default=5)
    parser.add_argument("--denoise-model", default="rectified_flow")
    parser.add_argument("--backbone", default="clip")
    parser.add_argument("--custom-img-size", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--dataset", default="MesaBimanual")

    # ---- Eval client passthroughs ----
    parser.add_argument("--eval-set-name", default="mesa_bimanual")
    parser.add_argument("--eval-split", default="overfit")
    parser.add_argument("--task-filter", nargs="+", default=["apple_tray_on"])
    parser.add_argument("--exp-name", default="3dfa_mesa")
    parser.add_argument("--variant-name", required=True,
                        help="Variant subdir; usually the run name (e.g. keypose-best).")
    parser.add_argument("--num-rollouts-per-task", type=int, default=10)
    parser.add_argument("--num-env-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--camera-names", nargs="+", default=DEFAULT_CAMERA_NAMES)
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--robots", nargs="+",
                        default=["ReverseMountedYam", "ReverseMountedYam"])
    parser.add_argument("--controller-type", default="osc_pose")
    parser.add_argument("--depth-transport", default="raw",
                        choices=["raw", "meters", "millimeters"])
    parser.add_argument("--state-keys", nargs="+", default=DEFAULT_STATE_KEYS)
    parser.add_argument("--video-out-path", default="experiments/vla_benchmark")
    parser.add_argument("--visualization-camera-name", default="egocentric")
    parser.add_argument("--replan-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)

    return parser.parse_args()


def _build_serve_cmd(python_exe: str, args: argparse.Namespace, port: int) -> list[str]:
    cmd = [
        python_exe,
        os.path.join(REPO_DIR, "serve_policy.py"),
        "--checkpoint", args.checkpoint,
        "--bimanual", args.bimanual,
        "--num_history", str(args.num_history),
        "--embedding_dim", str(args.embedding_dim),
        "--num_attn_heads", str(args.num_attn_heads),
        "--num_vis_instr_attn_layers", str(args.num_vis_instr_attn_layers),
        "--num_shared_attn_layers", str(args.num_shared_attn_layers),
        "--rotation_format", args.rotation_format,
        "--denoise_timesteps", str(args.denoise_timesteps),
        "--denoise_model", args.denoise_model,
        "--backbone", args.backbone,
        "--custom_img_size", str(args.custom_img_size),
        "--chunk_size", str(args.chunk_size),
        "--dataset", args.dataset,
        "--host", args.policy_host,
        "--port", str(port),
        "--exit_on_first_disconnect", "True",
        "--state_keys", *args.state_keys,
    ]
    if args.use_ema:
        cmd += ["--use_ema", "True"]
    return cmd


def _build_eval_cmd(args: argparse.Namespace, port: int) -> list[str]:
    shim = os.path.join(REPO_DIR, "scripts", "mesa", "run-vla-benchmark.sh")
    cmd = [
        "bash", shim,
        f"--eval-set-name={args.eval_set_name}",
        f"--eval-split={args.eval_split}",
        f"--exp-name={args.exp_name}",
        f"--variant-name={args.variant_name}",
        f"--num-rollouts-per-task={args.num_rollouts_per_task}",
        f"--num-env-workers={args.num_env_workers}",
        f"--controller-type={args.controller_type}",
        f"--seed={args.seed}",
        f"--port={port}",
        f"--host={args.policy_host}",
        f"--video-out-path={args.video_out_path}",
        f"--camera-height={args.camera_height}",
        f"--camera-width={args.camera_width}",
        f"--depth-transport={args.depth_transport}",
        f"--replan-steps={args.replan_steps}",
        f"--visualization-camera-name={args.visualization_camera_name}",
        "--camera-depths",
        "--camera-names", *args.camera_names,
        "--robots", *args.robots,
        "--state-keys", *args.state_keys,
        "--task-filter", *args.task_filter,
    ]
    if args.max_steps is not None:
        cmd.append(f"--max-steps={args.max_steps}")
    return cmd


async def main() -> int:
    args = _parse_args()

    summary_path = os.path.join(
        args.video_out_path,
        args.eval_set_name,
        args.exp_name,
        args.variant_name,
        "statistics",
        "final_summary.json",
    )
    if os.path.exists(summary_path):
        print(f"[launch] final summary already exists, skipping run: {summary_path}", flush=True)
        return 0

    port = args.policy_port if args.policy_port is not None else find_free_port()

    python_exe = os.path.join(args.policy_venv, "bin", "python")
    if not os.path.exists(python_exe):
        raise FileNotFoundError(
            f"Policy venv python missing: {python_exe} (use --policy-venv to override)"
        )

    serve_cmd = _build_serve_cmd(python_exe, args, port)
    eval_cmd = _build_eval_cmd(args, port)

    print(f"[launch] repo_dir   = {REPO_DIR}", flush=True)
    print(f"[launch] policy_venv= {args.policy_venv}", flush=True)
    print(f"[launch] policy port= {port}", flush=True)
    print(f"[launch] serve_cmd  = {' '.join(serve_cmd)}", flush=True)
    print(f"[launch] eval_cmd   = {' '.join(eval_cmd)}", flush=True)

    serve = Child("serve", serve_cmd)
    evalc = Child("eval", eval_cmd)
    children = [serve, evalc]

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _handle_stop(*_: object) -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_stop)
        except NotImplementedError:
            pass

    venv_extra_env = {
        "VIRTUAL_ENV": args.policy_venv,
        "PATH": f"{os.path.join(args.policy_venv, 'bin')}:{os.environ.get('PATH', '')}",
    }

    try:
        await start_child(serve, cwd=REPO_DIR, extra_env=venv_extra_env)
        await wait_port_listening(
            args.policy_host, port,
            timeout_s=args.server_startup_timeout,
            proc=serve.proc,
        )
        print(f"[launch] policy server listening on {args.policy_host}:{port}", flush=True)

        await start_child(evalc, cwd=REPO_DIR)

        waiters = [asyncio.create_task(c.proc.wait()) for c in children if c.proc]
        stop_task = asyncio.create_task(stop_event.wait())
        done, _pending = await asyncio.wait(
            waiters + [stop_task], return_when=asyncio.FIRST_COMPLETED
        )

        if stop_task in done:
            for c in children:
                await terminate_child(c)
            return 130

        rc_map = {c.name: (c.proc.returncode if c.proc else None) for c in children}
        print(f"[launch] return codes: {rc_map}", flush=True)
        failed = [name for name, rc in rc_map.items() if rc is None or rc != 0]
        for c in children:
            await terminate_child(c)
        return 1 if failed else 0
    finally:
        for c in children:
            await terminate_child(c)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
