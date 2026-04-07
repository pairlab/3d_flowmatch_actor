"""Websocket policy server for 3DFA mesa-bimanual checkpoints.

The server loads a `DenoiseActor3D` checkpoint trained via `main.py`, accepts
observations from `vla-benchmark`'s `eval_server_parallel.py` over the openpi
msgpack-numpy wire protocol, runs inference, and returns actions in the format
expected by the mesa `osc_pose` controller (`abs_ee_pose`).

"""

import argparse
import asyncio
import collections
import contextlib
import http
import json
import logging
import time
import traceback
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import websockets
import websockets.asyncio.server as _ws_server
import websockets.frames
from openpi_client import msgpack_numpy
from scipy.spatial.transform import Rotation

from data_processing.mesa_to_zarr import (
    CAMERA_ORDER,
    JAW_MAX,
    JAW_MIN,
    NUM_HANDS,
    ZFAR,
    ZNEAR,
)
from modeling.encoder.text import fetch_tokenizers
from modeling.policy import fetch_model_class
from utils.common_utils import str2bool
from utils.depth2cloud import fetch_depth2cloud


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Per-key flat dimensions for keys that may appear in `obs["state"]`.
# Camera intrinsic/extrinsic keys are matched by suffix below.
STATE_KEY_DIMS: Dict[str, int] = {
    "robot0_eef_pos": 3,
    "robot0_eef_quat": 4,
    "robot0_gripper_jaw_width": 1,
    "robot1_eef_pos": 3,
    "robot1_eef_quat": 4,
    "robot1_gripper_jaw_width": 1,
}
_SUFFIX_DIMS = {"_intrinsic": 9, "_extrinsic": 16}
_SUFFIX_SHAPES = {"_intrinsic": (3, 3), "_extrinsic": (4, 4)}


def _key_flat_dim_and_shape(key: str) -> Tuple[int, Tuple[int, ...]]:
    if key in STATE_KEY_DIMS:
        return STATE_KEY_DIMS[key], (STATE_KEY_DIMS[key],)
    for suffix, dim in _SUFFIX_DIMS.items():
        if key.endswith(suffix):
            return dim, _SUFFIX_SHAPES[suffix]
    raise ValueError(
        f"Unknown state key: {key!r}. Known keys: {list(STATE_KEY_DIMS.keys())} "
        f"or any *_intrinsic / *_extrinsic suffix."
    )


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip leading 'module.' added by DDP wrapping during training."""
    out = OrderedDict()
    for k, v in state_dict.items():
        out[k[len("module."):] if k.startswith("module.") else k] = v
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("3DFA mesa-bimanual policy websocket server")

    # --- Checkpoint ---
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pth checkpoint (last/best/intermN).")
    parser.add_argument("--use_ema", type=str2bool, default=False,
                        help="If True, load `ema_weight` instead of `weight`.")

    # --- Model architecture (must match training) ---
    parser.add_argument("--model_type", type=str, default="denoise3d")
    parser.add_argument("--bimanual", type=str2bool, default=True)
    parser.add_argument("--num_history", type=int, default=3)
    parser.add_argument("--embedding_dim", type=int, default=120)
    parser.add_argument("--num_attn_heads", type=int, default=8)
    parser.add_argument("--num_vis_instr_attn_layers", type=int, default=3)
    parser.add_argument("--num_shared_attn_layers", type=int, default=4)
    parser.add_argument("--fps_subsampling_factor", type=int, default=5)
    parser.add_argument("--rotation_format", type=str, default="quat_xyzw")
    parser.add_argument("--denoise_timesteps", type=int, default=5)
    parser.add_argument("--denoise_model", type=str, default="rectified_flow")
    parser.add_argument("--backbone", type=str, default="clip")
    parser.add_argument("--finetune_backbone", type=str2bool, default=False)
    parser.add_argument("--finetune_text_encoder", type=str2bool, default=False)
    parser.add_argument("--relative_action", type=str2bool, default=False)
    parser.add_argument("--lv2_batch_size", type=int, default=1)
    parser.add_argument("--custom_img_size", type=int, default=128)
    parser.add_argument("--chunk_size", type=int, default=1,
                        help="Length of returned action chunk per inference call.")

    # --- Dataset routing (controls preprocessor + depth2cloud) ---
    parser.add_argument("--dataset", type=str, default="MesaBimanual")

    # --- Server ---
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--exit_on_first_disconnect", type=str2bool, default=True)
    parser.add_argument("--state_keys", type=str, nargs="+",
                        default=[
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
                        ],
                        help="Order of keys in obs['state'] sent by the eval client. "
                             "Must match the eval client's --state-keys exactly.")
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def load_3dfa_policy(args: argparse.Namespace) -> Tuple[torch.nn.Module, Any]:
    """Build a `DenoiseActor3D` matching `args` and load weights from checkpoint.

    Returns the model (in eval mode, on the requested device) and a tokenizer.
    """
    model_cls = fetch_model_class(args.model_type)
    if model_cls is None:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    model = model_cls(
        backbone=args.backbone,
        finetune_backbone=args.finetune_backbone,
        finetune_text_encoder=args.finetune_text_encoder,
        num_vis_instr_attn_layers=args.num_vis_instr_attn_layers,
        fps_subsampling_factor=args.fps_subsampling_factor,
        embedding_dim=args.embedding_dim,
        num_attn_heads=args.num_attn_heads,
        nhist=args.num_history,
        nhand=2 if args.bimanual else 1,
        num_shared_attn_layers=args.num_shared_attn_layers,
        relative=args.relative_action,
        rotation_format=args.rotation_format,
        denoise_timesteps=args.denoise_timesteps,
        denoise_model=args.denoise_model,
        lv2_batch_size=args.lv2_batch_size,
    )

    logger.info("Loading checkpoint from %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    weight_key = "ema_weight" if args.use_ema else "weight"
    if weight_key not in ckpt or ckpt[weight_key] is None:
        raise KeyError(
            f"Checkpoint at {args.checkpoint} does not contain '{weight_key}'. "
            f"Available keys: {list(ckpt.keys())}"
        )
    state_dict = _strip_module_prefix(ckpt[weight_key])

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys when loading checkpoint (%d): %s", len(missing), missing)
    if unexpected:
        logger.warning("Unexpected keys when loading checkpoint (%d): %s", len(unexpected), unexpected)
    logger.info("workspace_normalizer (recovered from ckpt): %s",
                model.workspace_normalizer.detach().cpu().tolist())

    model = model.to(args.device)
    model.eval()

    tokenizer = fetch_tokenizers(args.backbone)
    if tokenizer is None:
        raise ValueError(f"No tokenizer registered for backbone={args.backbone}")
    return model, tokenizer


class Mesa3DFAPolicy:
    """Wrap a 3DFA `DenoiseActor3D` for mesa-bimanual websocket serving."""

    CAMERA_ORDER = CAMERA_ORDER  # ("egocentric", "robot0_eye_in_hand", "robot1_eye_in_hand")

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        state_keys: List[str],
        num_history: int,
        chunk_size: int,
        custom_img_size: int,
        dataset: str,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.state_keys = list(state_keys)
        self.num_history = int(num_history)
        self.chunk_size = int(chunk_size)
        self.custom_img_size = int(custom_img_size)
        self.device = device

        # Build state slice plan from --state_keys (mirrors eval client side).
        self.state_slices: Dict[str, Tuple[int, int]] = {}
        self.state_shapes: Dict[str, Tuple[int, ...]] = {}
        offset = 0
        for key in self.state_keys:
            flat_dim, shape = _key_flat_dim_and_shape(key)
            self.state_slices[key] = (offset, offset + flat_dim)
            self.state_shapes[key] = shape
            offset += flat_dim
        self.expected_state_dim = offset
        logger.info("State plan (%d dims): %s", self.expected_state_dim, self.state_slices)

        for cam in self.CAMERA_ORDER:
            if f"{cam}_intrinsic" not in self.state_slices:
                raise ValueError(
                    f"--state_keys must include '{cam}_intrinsic' for the depth → "
                    f"point cloud unprojection."
                )
            if f"{cam}_extrinsic" not in self.state_slices:
                raise ValueError(
                    f"--state_keys must include '{cam}_extrinsic' for the depth → "
                    f"point cloud unprojection."
                )

        # depth2cloud at the same resolution used during training.
        self.depth2cloud = fetch_depth2cloud(
            dataset, img_size=(self.custom_img_size, self.custom_img_size)
        )
        if self.depth2cloud is None:
            raise ValueError(f"No depth2cloud routed for dataset={dataset!r}")

        # Per-server state.
        self._proprio_history: collections.deque = collections.deque(maxlen=self.num_history)
        self._cached_prompt: str | None = None
        self._cached_instr_tokens: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Episode-state helpers
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._proprio_history.clear()
        self._cached_prompt = None
        self._cached_instr_tokens = None

    def _maybe_reset_history(self, current_proprio: np.ndarray) -> None:
        """Heuristic episode-boundary reset.

        The eval server keeps the same websocket connection across episodes
        and never signals reset to the policy. If the new proprio jumps far
        from the last (>0.5 m on either hand) we treat it as a new episode and
        flush history. Same-position no-ops are common, so we use a coarse
        threshold rather than a strict equality check.
        """
        if not self._proprio_history:
            return
        last = self._proprio_history[-1]  # (nhand, 8)
        delta = np.linalg.norm(current_proprio[..., :3] - last[..., :3], axis=-1)
        if float(delta.max()) > 0.5:
            logger.info("Detected proprio jump (%.3f m); resetting history.", float(delta.max()))
            self._proprio_history.clear()

    # ------------------------------------------------------------------
    # Observation preprocessing
    # ------------------------------------------------------------------
    def _slice_state(self, state_flat: np.ndarray, key: str) -> np.ndarray:
        start, end = self.state_slices[key]
        shape = self.state_shapes[key]
        return state_flat[start:end].reshape(*shape).astype(np.float32)

    def _build_proprio(self, state_flat: np.ndarray) -> torch.Tensor:
        """Build proprio of shape (1, num_history, 2, 8) from a flat state vector.

        Per-hand 8D layout: [eef_pos(3), eef_quat_xyzw(4), gripper_norm(1)],
        which exactly matches the layout written by `mesa_to_zarr.py`.
        """
        per_hand = []
        for hand_idx in range(NUM_HANDS):
            eef_pos = self._slice_state(state_flat, f"robot{hand_idx}_eef_pos")  # (3,)
            eef_quat = self._slice_state(state_flat, f"robot{hand_idx}_eef_quat")  # (4,)
            jaw = self._slice_state(state_flat, f"robot{hand_idx}_gripper_jaw_width")  # (1,)
            grip_norm = np.clip((jaw - JAW_MIN) / (JAW_MAX - JAW_MIN), 0.0, 1.0)
            per_hand.append(
                np.concatenate([eef_pos, eef_quat, grip_norm.astype(np.float32)], axis=-1)
            )
        current = np.stack(per_hand, axis=0).astype(np.float32)  # (2, 8)

        self._maybe_reset_history(current)
        self._proprio_history.append(current)

        # Replicate-pad with the earliest available frame to fill the window.
        history = list(self._proprio_history)
        while len(history) < self.num_history:
            history.insert(0, history[0])
        proprio = np.stack(history, axis=0)  # (num_history, 2, 8)
        return torch.from_numpy(proprio).unsqueeze(0).to(self.device, non_blocking=True)

    def _build_rgbs(self, images: Dict[str, Any]) -> torch.Tensor:
        per_cam = []
        for cam in self.CAMERA_ORDER:
            if cam not in images:
                raise KeyError(
                    f"obs['images'] missing camera {cam!r}. Got: {list(images.keys())}"
                )
            img = np.asarray(images[cam])
            if img.shape[-1] != 3 or img.ndim != 3:
                raise ValueError(
                    f"Camera {cam!r}: expected (H, W, 3), got shape {img.shape}"
                )
            chw = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
            per_cam.append(chw)
        rgbs = np.stack(per_cam, axis=0)  # (3, 3, H, W)
        return torch.from_numpy(rgbs).unsqueeze(0).to(self.device, non_blocking=True)

    def _build_pcds(
        self, images: Dict[str, Any], state_flat: np.ndarray
    ) -> torch.Tensor:
        depths = []
        intrinsics = []
        extrinsics = []
        for cam in self.CAMERA_ORDER:
            depth_key = f"{cam}_depth"
            if depth_key not in images:
                raise KeyError(
                    f"obs['images'] missing depth {depth_key!r}. Did you pass "
                    f"--camera-depths and include this camera in --camera-names?"
                )
            raw = np.asarray(images[depth_key], dtype=np.float32)
            if raw.ndim == 3 and raw.shape[-1] == 1:
                raw = raw[..., 0]
            if raw.ndim != 2:
                raise ValueError(
                    f"Depth {depth_key!r}: expected (H, W) or (H, W, 1), got {raw.shape}"
                )
            # The eval client is launched with --depth-transport raw, so we get
            # the MuJoCo z-buffer and linearize with the same constants used in
            # data_processing/mesa_to_zarr.py.
            depth_m = ZNEAR / (1.0 - raw * (1.0 - ZNEAR / ZFAR))
            depths.append(depth_m.astype(np.float32))
            intrinsics.append(self._slice_state(state_flat, f"{cam}_intrinsic"))
            extrinsics.append(self._slice_state(state_flat, f"{cam}_extrinsic"))

        depth_t = torch.from_numpy(np.stack(depths, axis=0)).unsqueeze(0)         # (1, 3, H, W)
        intr_t = torch.from_numpy(np.stack(intrinsics, axis=0)).unsqueeze(0)      # (1, 3, 3, 3)
        extr_t = torch.from_numpy(np.stack(extrinsics, axis=0)).unsqueeze(0)      # (1, 3, 4, 4)

        depth_t = depth_t.to(self.device, non_blocking=True)
        intr_t = intr_t.to(self.device, non_blocking=True)
        extr_t = extr_t.to(self.device, non_blocking=True)
        pcds = self.depth2cloud(depth_t, extr_t, intr_t)  # (1, 3, 3, H, W)
        return pcds.float()

    def _build_instruction(self, prompt: str) -> torch.Tensor:
        if prompt != self._cached_prompt or self._cached_instr_tokens is None:
            tokens = self.tokenizer([prompt]).to(self.device, non_blocking=True)
            self._cached_prompt = prompt
            self._cached_instr_tokens = tokens
        return self._cached_instr_tokens

    def _preprocess_obs(self, obs: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        if "images" not in obs or "state" not in obs or "prompt" not in obs:
            raise ValueError(
                f"Observation missing required keys; got: {sorted(obs.keys())}"
            )

        state_flat = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        if state_flat.size != self.expected_state_dim:
            raise ValueError(
                f"obs['state'] has {state_flat.size} dims but server expects "
                f"{self.expected_state_dim} (--state_keys order: {self.state_keys})."
            )

        proprio = self._build_proprio(state_flat)
        rgbs = self._build_rgbs(obs["images"])
        pcds = self._build_pcds(obs["images"], state_flat)
        instr_tokens = self._build_instruction(str(obs["prompt"]))

        # zeros mask of shape (B=1, T=chunk_size, nhand=2). Mirrors
        # RLBenchTrainTester.prepare_batch().
        mask = torch.zeros(
            (1, self.chunk_size, 2), dtype=torch.bool, device=self.device
        )
        return mask, rgbs, pcds, instr_tokens, proprio

    # ------------------------------------------------------------------
    # Inference + action postprocessing
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.monotonic()
        mask, rgbs, pcds, instr_tokens, proprio = self._preprocess_obs(obs)

        traj = self.model(
            None,           # gt_trajectory
            mask,           # trajectory_mask
            rgbs,           # rgb3d
            None,           # rgb2d
            pcds,           # pcd
            instr_tokens,   # instruction
            proprio,        # proprio
            run_inference=True,
        )  # (1, T, 2, 8) — [x, y, z, qx, qy, qz, qw, grip_sigmoid]

        actions = self._postprocess_actions(traj)
        infer_ms = (time.monotonic() - t0) * 1000.0
        return {"actions": actions.tolist(), "server_timing": {"infer_ms": infer_ms}}

    def _postprocess_actions(self, traj: torch.Tensor) -> np.ndarray:
        """Convert model output (1, T, 2, 8) → env-format (T, 14).

        Mesa env layout per timestep:
            [pos0(3), aa0(3), grip0(1), pos1(3), aa1(3), grip1(1)]

        Hand index 0 corresponds to robot0 (mesa_to_zarr writes hand_idx 0 from
        `robot0_*` keys); the eval env consumes the same convention as the
        source `abs_actions_ee_pose`, so no per-hand reorder is needed.
        """
        traj_np = traj.detach().to(torch.float32).cpu().numpy()  # (1, T, 2, 8)
        if traj_np.shape[0] != 1 or traj_np.shape[2] != 2 or traj_np.shape[3] != 8:
            raise ValueError(
                f"Unexpected model output shape: {traj_np.shape}; expected (1, T, 2, 8)"
            )
        T = traj_np.shape[1]

        pos = traj_np[0, :, :, 0:3]                     # (T, 2, 3)
        quat_xyzw = traj_np[0, :, :, 3:7]               # (T, 2, 4)
        grip_sigmoid = traj_np[0, :, :, 7]              # (T, 2)

        # quat → axis-angle, vectorized over T*2.
        quats_flat = quat_xyzw.reshape(-1, 4)
        # Renormalize to suppress drift before scipy's strict checks.
        quats_flat = quats_flat / np.clip(
            np.linalg.norm(quats_flat, axis=-1, keepdims=True), 1e-8, None
        )
        aa = Rotation.from_quat(quats_flat).as_rotvec().reshape(T, 2, 3).astype(np.float32)

        # sigmoid output ∈ [0, 1] → discrete {-1, +1} expected by mesa env.
        grip_env = (2.0 * (grip_sigmoid > 0.5).astype(np.float32) - 1.0).reshape(T, 2, 1)

        per_hand = np.concatenate([pos, aa, grip_env], axis=-1)  # (T, 2, 7)
        return per_hand.reshape(T, 14).astype(np.float32)


# ---------------------------------------------------------------------------
# Websocket server (mirrors openpi-client / imitation reference structure).
# ---------------------------------------------------------------------------
class WebsocketPolicyServer:
    """Serve a policy over a websocket using openpi msgpack-numpy framing."""

    def __init__(
        self,
        policy: Mesa3DFAPolicy,
        *,
        host: str,
        port: int,
        metadata: Dict[str, Any] | None = None,
        exit_on_first_disconnect: bool = False,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._exit_on_first_disconnect = exit_on_first_disconnect
        self._server: _ws_server.Server | None = None
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        try:
            async with _ws_server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                process_request=_health_check,
            ) as server:
                self._server = server
                await server.serve_forever()
        except asyncio.CancelledError:
            logger.info("Server cancelled during shutdown; exiting cleanly.")
        finally:
            if self._server is not None:
                await self._server.wait_closed()
                self._server = None

    async def _handler(self, websocket: _ws_server.ServerConnection) -> None:
        logger.info("Connection opened from %s", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        # Send metadata frame first; openpi clients consume this on connect.
        await websocket.send(packer.pack(self._metadata))
        prev_total_time: float | None = None
        while True:
            try:
                start = time.monotonic()
                message = await websocket.recv()
                if not isinstance(message, (bytes, bytearray)):
                    raise TypeError(
                        f"Unexpected message type from client: {type(message).__name__}"
                    )
                obs = msgpack_numpy.unpackb(message)
                action = self._policy.infer(obs)
                action.setdefault("server_timing", {})
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0
                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start
            except websockets.ConnectionClosed:
                logger.info("Connection closed from %s", websocket.remote_address)
                if self._exit_on_first_disconnect and self._server is not None:
                    logger.info("Exiting server due to first disconnect.")
                    self._server.close()
                return
            except Exception:
                tb = traceback.format_exc()
                logger.exception("Policy server error while handling request.")
                with contextlib.suppress(Exception):
                    await websocket.send(json.dumps({"error": tb}))
                with contextlib.suppress(Exception):
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error.",
                    )
                raise


def _health_check(
    connection: _ws_server.ServerConnection, request: _ws_server.Request
) -> _ws_server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def main() -> None:
    args = parse_args()
    logger.info("Arguments: %s", args)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    model, tokenizer = load_3dfa_policy(args)
    policy = Mesa3DFAPolicy(
        model=model,
        tokenizer=tokenizer,
        state_keys=args.state_keys,
        num_history=args.num_history,
        chunk_size=args.chunk_size,
        custom_img_size=args.custom_img_size,
        dataset=args.dataset,
        device=args.device,
    )
    metadata = {
        "server": "3dfa_mesa_bimanual",
        "checkpoint": args.checkpoint,
        "num_history": args.num_history,
        "chunk_size": args.chunk_size,
        "custom_img_size": args.custom_img_size,
        "state_keys": args.state_keys,
        "expected_state_dim": policy.expected_state_dim,
    }
    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
        exit_on_first_disconnect=args.exit_on_first_disconnect,
    )
    logger.info("Starting policy server on %s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
