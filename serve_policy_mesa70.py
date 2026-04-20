"""Websocket policy server for 3DFA SmolVLA on Mesa-70 (single-arm, no depth).

Accepts observations from vla-benchmark's eval_server_parallel.py over the
openpi msgpack-numpy wire protocol.  Camera depth is NOT required — point
clouds are synthesised from a normalized pixel grid (matching training).

Expected obs layout
-------------------
obs["images"]
    "leftshoulder"          : (H, W, 3) uint8
    "rightshoulder"         : (H, W, 3) uint8
    "robot0_eye_in_hand"    : (H, W, 3) uint8

obs["state"]   flat float32 vector, in --state_keys order
obs["prompt"]  str

Default --state_keys:
    robot0_eef_pos (3), robot0_eef_quat (4), robot0_gripper_jaw_width (1)
    Total: 8 dims

Action returned to env (T, 7):
    [pos(3), axis_angle(3), gripper_binary(1)]  per timestep

Usage example
-------------
python serve_policy_mesa70.py \\
    --checkpoint /path/to/last.pth \\
    --model_type smolvla_prefix_kv \\
    --num_history 3 \\
    --chunk_size 1 \\
    --port 8000
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

from modeling.policy import fetch_model_class
from utils.common_utils import str2bool
from utils.depth2cloud.mesa70 import Mesa70PixelGridCloud


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CAMERA_ORDER = ("leftshoulder", "rightshoulder", "robot0_eye_in_hand")

JAW_MIN = 0.0
JAW_MAX = 0.121

STATE_KEY_DIMS: Dict[str, int] = {
    "robot0_eef_pos": 3,
    "robot0_eef_quat": 4,
    "robot0_gripper_jaw_width": 1,
}


def _strip_module_prefix(sd):
    out = OrderedDict()
    for k, v in sd.items():
        out[k[len("module."):] if k.startswith("module.") else k] = v
    return out


def parse_args():
    parser = argparse.ArgumentParser("3DFA mesa-70 SmolVLA policy server")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--use_ema", type=str2bool, default=False)

    # Model architecture (must match training)
    parser.add_argument("--model_type", type=str, default="smolvla_prefix_kv")
    parser.add_argument("--num_history", type=int, default=3)
    parser.add_argument("--embedding_dim", type=int, default=120)
    parser.add_argument("--num_attn_heads", type=int, default=8)
    parser.add_argument("--num_shared_attn_layers", type=int, default=4)
    parser.add_argument("--fps_subsampling_factor", type=int, default=5)
    parser.add_argument("--rotation_format", type=str, default="quat_xyzw")
    parser.add_argument("--denoise_timesteps", type=int, default=5)
    parser.add_argument("--denoise_model", type=str, default="rectified_flow")
    parser.add_argument("--relative_action", type=str2bool, default=False)
    parser.add_argument("--lv2_batch_size", type=int, default=1)
    parser.add_argument("--custom_img_size", type=int, default=128)
    parser.add_argument("--chunk_size", type=int, default=1)

    # SmolVLA-specific
    parser.add_argument("--smolvlm_model_name", type=str,
                        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--smolvlm_local_files_only", type=str2bool, default=True)
    parser.add_argument("--smolvlm_freeze_vision_tower", type=str2bool, default=True)
    parser.add_argument("--smolvlm_freeze_connector", type=str2bool, default=True)
    parser.add_argument("--smolvlm_freeze_text_model", type=str2bool, default=True)
    parser.add_argument("--smolvlm_freeze_text_embeddings", type=str2bool, default=True)
    parser.add_argument("--smolvlm_tokenizer_max_length", type=int, default=48)
    parser.add_argument("--smolvlm_append_state_tokens", type=str2bool, default=False)
    parser.add_argument("--smolvlm_state_dim", type=int, default=0)
    parser.add_argument("--smolvlm_num_state_tokens", type=int, default=1)

    # Server
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--exit_on_first_disconnect", type=str2bool, default=True)
    parser.add_argument("--state_keys", type=str, nargs="+",
                        default=["robot0_eef_pos", "robot0_eef_quat",
                                 "robot0_gripper_jaw_width"])
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def load_policy(args) -> torch.nn.Module:
    model_cls = fetch_model_class(args.model_type)
    if model_cls is None:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    model = model_cls(
        smolvlm_model_name=args.smolvlm_model_name,
        smolvlm_local_files_only=args.smolvlm_local_files_only,
        smolvlm_freeze_vision_tower=args.smolvlm_freeze_vision_tower,
        smolvlm_freeze_connector=args.smolvlm_freeze_connector,
        smolvlm_freeze_text_model=args.smolvlm_freeze_text_model,
        smolvlm_freeze_text_embeddings=args.smolvlm_freeze_text_embeddings,
        smolvlm_tokenizer_max_length=args.smolvlm_tokenizer_max_length,
        smolvlm_append_state_tokens=args.smolvlm_append_state_tokens,
        smolvlm_state_dim=args.smolvlm_state_dim,
        smolvlm_num_state_tokens=args.smolvlm_num_state_tokens,
        fps_subsampling_factor=args.fps_subsampling_factor,
        embedding_dim=args.embedding_dim,
        num_attn_heads=args.num_attn_heads,
        nhist=args.num_history,
        nhand=1,
        num_shared_attn_layers=args.num_shared_attn_layers,
        relative=args.relative_action,
        rotation_format=args.rotation_format,
        denoise_timesteps=args.denoise_timesteps,
        denoise_model=args.denoise_model,
        lv2_batch_size=args.lv2_batch_size,
    )

    logger.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    weight_key = "ema_weight" if args.use_ema else "weight"
    if weight_key not in ckpt or ckpt[weight_key] is None:
        raise KeyError(f"Checkpoint missing '{weight_key}'. Keys: {list(ckpt.keys())}")
    sd = _strip_module_prefix(ckpt[weight_key])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("Missing keys (%d): %s", len(missing), missing[:5])
    if unexpected:
        logger.warning("Unexpected keys (%d): %s", len(unexpected), unexpected[:5])
    logger.info("workspace_normalizer: %s",
                model.workspace_normalizer.detach().cpu().tolist())

    model = model.to(args.device).eval()
    return model


class Mesa70Policy:
    """Single-arm Mesa-70 policy with pixel-grid point clouds."""

    def __init__(self, model, state_keys, num_history, chunk_size,
                 custom_img_size, device="cuda"):
        self.model = model
        self.state_keys = list(state_keys)
        self.num_history = int(num_history)
        self.chunk_size = int(chunk_size)
        self.img_size = int(custom_img_size)
        self.device = device

        # Build state offset map
        self.state_slices: Dict[str, Tuple[int, int]] = {}
        offset = 0
        for key in self.state_keys:
            dim = STATE_KEY_DIMS.get(key)
            if dim is None:
                raise ValueError(f"Unknown state key: {key!r}. Known: {list(STATE_KEY_DIMS)}")
            self.state_slices[key] = (offset, offset + dim)
            offset += dim
        self.expected_state_dim = offset
        logger.info("State plan (%d dims): %s", self.expected_state_dim, self.state_slices)

        # Pixel-grid pcd (no depth needed)
        self.pcd_gen = Mesa70PixelGridCloud((self.img_size, self.img_size))

        self._proprio_history: collections.deque = collections.deque(maxlen=num_history)
        self._cached_prompt: str | None = None

    def reset(self):
        self._proprio_history.clear()
        self._cached_prompt = None

    def _slice_state(self, flat, key):
        s, e = self.state_slices[key]
        return flat[s:e].astype(np.float32)

    def _maybe_reset_history(self, current_eef_pos):
        if not self._proprio_history:
            return
        last_pos = self._proprio_history[-1][:3]
        if float(np.linalg.norm(current_eef_pos - last_pos)) > 0.5:
            logger.info("Proprio jump detected; resetting history.")
            self._proprio_history.clear()

    def _build_proprio(self, flat):
        """→ (1, num_history, 1, 8) tensor on device."""
        eef_pos = self._slice_state(flat, "robot0_eef_pos")   # (3,)
        eef_quat = self._slice_state(flat, "robot0_eef_quat") # (4,)
        jaw = self._slice_state(flat, "robot0_gripper_jaw_width")  # (1,)
        grip_norm = np.clip((jaw - JAW_MIN) / (JAW_MAX - JAW_MIN), 0.0, 1.0)
        current = np.concatenate([eef_pos, eef_quat, grip_norm]).astype(np.float32)  # (8,)

        self._maybe_reset_history(eef_pos)
        self._proprio_history.append(current)

        history = list(self._proprio_history)
        while len(history) < self.num_history:
            history.insert(0, history[0])

        proprio = np.stack(history, axis=0)[:, None, :]  # (nhist, 1, 8)
        return torch.from_numpy(proprio).unsqueeze(0).to(self.device)

    def _build_rgbs(self, images):
        """→ (1, 3, 3, img_size, img_size) float32 in [0,1]."""
        per_cam = []
        for cam in CAMERA_ORDER:
            img = np.asarray(images[cam])  # (H, W, 3) uint8
            if img.shape[:2] != (self.img_size, self.img_size):
                from PIL import Image as _PIL
                img = np.array(
                    _PIL.fromarray(img).resize((self.img_size, self.img_size), _PIL.BILINEAR)
                )
            chw = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            per_cam.append(chw)
        rgbs = np.stack(per_cam, axis=0)  # (3, 3, H, W)
        return torch.from_numpy(rgbs).unsqueeze(0).to(self.device)

    def _build_pcds(self, rgbs_tensor):
        """Pixel-grid pcd → (1, 3, 3, img_size, img_size)."""
        b, nc, _, h, w = rgbs_tensor.shape
        dummy_depth = torch.zeros(b, nc, h, w, device=self.device)
        dummy_ext = torch.zeros(b, nc, 4, 4, device=self.device)
        dummy_intr = torch.zeros(b, nc, 3, 3, device=self.device)
        return self.pcd_gen(dummy_depth, dummy_ext, dummy_intr).float()

    @torch.inference_mode()
    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.monotonic()
        flat = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        if flat.size != self.expected_state_dim:
            raise ValueError(
                f"obs['state'] has {flat.size} dims; expected {self.expected_state_dim}"
            )

        proprio = self._build_proprio(flat)
        rgbs = self._build_rgbs(obs["images"])
        pcds = self._build_pcds(rgbs)
        prompt = str(obs["prompt"])

        mask = torch.zeros((1, self.chunk_size, 1), dtype=torch.bool, device=self.device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            traj = self.model(
                None,      # gt_trajectory
                mask,      # trajectory_mask
                rgbs,      # rgb3d  (1, 3, 3, H, W)
                None,      # rgb2d
                pcds,      # pcd    (1, 3, 3, H, W)
                [prompt],  # instruction — raw string, SmolVLM tokenises internally
                proprio,   # (1, nhist, 1, 8)
                run_inference=True,
            )  # (1, T, 1, 8)

        actions = self._postprocess_actions(traj)
        return {"actions": actions.tolist(),
                "server_timing": {"infer_ms": (time.monotonic() - t0) * 1e3}}

    def _postprocess_actions(self, traj: torch.Tensor) -> np.ndarray:
        """(1, T, 1, 8) → (T, 7): [pos(3), axis_angle(3), grip_binary(1)]."""
        arr = traj.detach().float().cpu().numpy()  # (1, T, 1, 8)
        T = arr.shape[1]
        pos = arr[0, :, 0, :3]           # (T, 3)
        quat_xyzw = arr[0, :, 0, 3:7]   # (T, 4)
        grip_sig = arr[0, :, 0, 7]      # (T,)

        # quat → axis-angle
        q = quat_xyzw / np.clip(np.linalg.norm(quat_xyzw, axis=-1, keepdims=True), 1e-8, None)
        aa = Rotation.from_quat(q).as_rotvec().astype(np.float32)  # (T, 3)

        # sigmoid ∈ [0, 1] → {-1, +1}
        grip = (2.0 * (grip_sig > 0.5).astype(np.float32) - 1.0).reshape(T, 1)

        return np.concatenate([pos, aa, grip], axis=-1).astype(np.float32)  # (T, 7)


# ---------------------------------------------------------------------------
# WebSocket server (same framing as serve_policy.py)
# ---------------------------------------------------------------------------

class WebsocketPolicyServer:
    def __init__(self, policy, *, host, port, metadata=None,
                 exit_on_first_disconnect=False):
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._exit_on_first_disconnect = exit_on_first_disconnect
        self._server = None

    def serve_forever(self):
        asyncio.run(self.run())

    async def run(self):
        try:
            async with _ws_server.serve(
                self._handler, self._host, self._port,
                compression=None, max_size=None,
                process_request=_health_check,
            ) as server:
                self._server = server
                await server.serve_forever()
        except asyncio.CancelledError:
            logger.info("Server cancelled.")
        finally:
            if self._server:
                await self._server.wait_closed()

    async def _handler(self, ws):
        logger.info("Connection from %s", ws.remote_address)
        packer = msgpack_numpy.Packer()
        await ws.send(packer.pack(self._metadata))
        prev_t = None
        while True:
            try:
                start = time.monotonic()
                msg = await ws.recv()
                obs = msgpack_numpy.unpackb(msg)
                result = self._policy.infer(obs)
                result.setdefault("server_timing", {})
                if prev_t is not None:
                    result["server_timing"]["prev_total_ms"] = prev_t * 1e3
                await ws.send(packer.pack(result))
                prev_t = time.monotonic() - start
            except websockets.ConnectionClosed:
                logger.info("Connection closed from %s", ws.remote_address)
                if self._exit_on_first_disconnect and self._server:
                    self._server.close()
                return
            except Exception:
                tb = traceback.format_exc()
                logger.exception("Error handling request.")
                with contextlib.suppress(Exception):
                    await ws.send(json.dumps({"error": tb}))
                with contextlib.suppress(Exception):
                    await ws.close(code=websockets.frames.CloseCode.INTERNAL_ERROR,
                                   reason="Internal server error.")
                raise


def _health_check(conn, req):
    if req.path == "/healthz":
        return conn.respond(http.HTTPStatus.OK, "OK\n")
    return None


def main():
    args = parse_args()
    logger.info("Args: %s", args)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    model = load_policy(args)
    policy = Mesa70Policy(
        model=model,
        state_keys=args.state_keys,
        num_history=args.num_history,
        chunk_size=args.chunk_size,
        custom_img_size=args.custom_img_size,
        device=args.device,
    )
    metadata = {
        "server": "3dfa_mesa70_smolvla",
        "checkpoint": args.checkpoint,
        "num_history": args.num_history,
        "chunk_size": args.chunk_size,
        "custom_img_size": args.custom_img_size,
        "state_keys": args.state_keys,
        "expected_state_dim": policy.expected_state_dim,
        "cameras": list(CAMERA_ORDER),
    }
    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
        exit_on_first_disconnect=args.exit_on_first_disconnect,
    )
    logger.info("Starting server on %s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
