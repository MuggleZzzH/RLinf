#!/usr/bin/env python3
"""Dexmal real-robot eval bridge.

This script connects a Dexmal robot gateway to a locally loaded RLinf/OpenPI policy:
1. poll robot observations from the gateway,
2. run policy inference locally on the server,
3. clip the full 50-step action chunk for safety,
4. send the chunk back to the robot for execution.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

DEFAULT_PROMPT = "fold the towel and place it into the output area"
MODEL_MODE = 1
STOP_MODE = 0
START_RECORD_MODE = 10
ACTION_HORIZON = 50
ACTION_DIM = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dexmal real-robot eval bridge.")
    parser.add_argument("--robot-host", default="127.0.0.1", help="Dexmal robot gateway host.")
    parser.add_argument("--robot-port", type=int, default=8000, help="Dexmal robot gateway port.")
    parser.add_argument("--request-timeout", type=float, default=5.0, help="HTTP timeout in seconds.")
    parser.add_argument("--poll-hz", type=float, default=10.0, help="Polling rate while waiting for observations.")

    parser.add_argument("--config-name", default="pi0_dexmal_aloha", help="RLinf OpenPI config name.")
    parser.add_argument("--checkpoint-dir", default=None, help="Trained checkpoint directory.")
    parser.add_argument("--assets-dir", default=None, help="Directory containing <asset_id>/norm_stats.json.")
    parser.add_argument("--repo-id", default=None, help="Dataset repo id or local dataset path.")
    parser.add_argument("--asset-id", default=None, help="Asset id used to load norm_stats.json.")
    parser.add_argument("--pytorch-device", default=None, help='PyTorch device, e.g. "cuda:0" or "cpu".')
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt sent to the policy.")

    parser.add_argument("--action-horizon", type=int, default=ACTION_HORIZON, help="Action chunk length to execute.")
    parser.add_argument("--action-scale", type=float, default=0.2, help="Action interpolation factor toward each model target.")
    parser.add_argument("--max-joint-delta", type=float, default=0.03, help="Per-step joint delta limit in radians.")
    parser.add_argument("--max-gripper-delta", type=float, default=0.005, help="Per-step gripper delta limit.")

    parser.add_argument("--set-model-mode-on-start", action="store_true", help="Switch robot to model mode on startup.")
    parser.add_argument("--set-stop-mode-on-exit", action="store_true", help="Switch robot to stop mode on exit.")
    parser.add_argument("--start-record-on-start", action="store_true", help="Send start-record before switching to model mode.")
    parser.add_argument(
        "--start-record-wait-seconds",
        type=float,
        default=3.0,
        help="Seconds to wait after start-record before enabling model mode.",
    )
    parser.add_argument("--shadow", action="store_true", help="Validate and send dry-run chunks without real execution.")
    parser.add_argument("--max-chunks", type=int, default=0, help="Maximum number of chunks to execute. 0 means unlimited.")
    parser.add_argument(
        "--log-dir",
        default="./eval_logs",
        help="Directory to write bridge chunk logs and summary.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name. Defaults to dexmal_eval_<timestamp>.",
    )

    parser.add_argument(
        "--mock-policy",
        choices=("zeros", "hold-state"),
        default=None,
        help="Serve a mock policy instead of loading a checkpoint.",
    )
    return parser.parse_args()


class RobotGatewayClient:
    def __init__(self, host: str, port: int, timeout: float):
        import requests

        self._base_url = f"http://{host}:{port}"
        self._timeout = timeout
        self._session = requests.Session()

    def get_mode(self) -> int | None:
        response = self._session.get(f"{self._base_url}/mode", timeout=self._timeout)
        response.raise_for_status()
        return response.json().get("data")

    def set_mode(self, mode: int) -> dict[str, Any]:
        response = self._session.post(
            f"{self._base_url}/mode",
            json={"collect_mode": mode},
            timeout=self._timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Failed to set mode {mode}: {response.text}")
        return response.json()

    def get_infer_state(self) -> dict[str, Any] | None:
        response = self._session.get(f"{self._base_url}/infer_state", timeout=self._timeout)
        response.raise_for_status()
        payload = pickle.loads(response.content)
        if not payload:
            return None
        return payload

    def post_action_chunk(
        self,
        *,
        obs_id: int | None,
        actions: np.ndarray,
        dry_run: bool,
        action_source: str,
    ) -> dict[str, Any]:
        response = self._session.post(
            f"{self._base_url}/infer_action",
            json={
                "obs_id": obs_id,
                "action_horizon": int(actions.shape[0]),
                "action_source": action_source,
                "client_timestamp": time.time(),
                "dry_run": dry_run,
                "action_list": actions.tolist(),
            },
            timeout=self._timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Failed to post action chunk: {response.text}")
        return response.json()


class JsonlLogger:
    def __init__(self, root: Path, run_name: str):
        self.run_dir = root / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._chunk_log_path = self.run_dir / "chunks.jsonl"
        self._summary_path = self.run_dir / "summary.json"

    def write_chunk(self, record: dict[str, Any]) -> None:
        with self._chunk_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_summary(self, summary: dict[str, Any]) -> None:
        self._summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def create_policy(args: argparse.Namespace):
    from toolkits.serve_dexmal_openpi import _MockPolicy, _create_checkpoint_policy

    if args.mock_policy is not None:
        return _MockPolicy(
            mode=args.mock_policy,
            action_horizon=args.action_horizon,
            action_dim=ACTION_DIM,
        )

    if args.checkpoint_dir is None:
        raise ValueError("--checkpoint-dir is required unless --mock-policy is used.")

    policy_args = SimpleNamespace(
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        assets_dir=args.assets_dir,
        repo_id=args.repo_id,
        asset_id=args.asset_id,
        host="0.0.0.0",
        port=None,
        default_prompt=args.prompt,
        pytorch_device=args.pytorch_device,
        metadata_json=None,
        record=False,
        use_repack_inputs=False,
        mock_policy=None,
        mock_action_horizon=args.action_horizon,
        mock_action_dim=ACTION_DIM,
    )
    return _create_checkpoint_policy(policy_args)


def decode_image(image_bytes: bytes) -> np.ndarray:
    try:
        import cv2

        frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode gateway image bytes with OpenCV.")
        return np.ascontiguousarray(frame)
    except ModuleNotFoundError:
        pass

    try:
        from PIL import Image

        with Image.open(BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            rgb = np.asarray(image, dtype=np.uint8)
        return np.ascontiguousarray(rgb[:, :, ::-1])
    except ModuleNotFoundError as exc:
        raise RuntimeError("decode_image requires either opencv-python or Pillow to be installed.") from exc


def build_observation(payload: dict[str, Any], prompt: str) -> tuple[int | None, np.ndarray, dict[str, Any]]:
    state = payload.get("state_14d", payload.get("action"))
    if state is None:
        raise ValueError("Gateway payload missing state_14d.")

    state = np.asarray(state, dtype=np.float32)
    if state.shape != (ACTION_DIM,):
        raise ValueError(f"Expected 14D state, got {state.shape}")

    images = payload.get("images", {})
    required = ("high", "left_hand", "right_hand")
    if any(key not in images for key in required):
        raise ValueError(f"Gateway payload missing images: expected {required}")

    observation = {
        "state": state,
        "images": {
            "cam_high": decode_image(images["high"]),
            "cam_left_wrist": decode_image(images["left_hand"]),
            "cam_right_wrist": decode_image(images["right_hand"]),
        },
        "prompt": prompt,
    }
    return payload.get("obs_id"), state, observation


def clip_action_chunk(
    current_state: np.ndarray,
    raw_actions: np.ndarray,
    *,
    action_scale: float,
    max_joint_delta: float,
    max_gripper_delta: float,
) -> np.ndarray:
    if current_state.shape != (ACTION_DIM,):
        raise ValueError(f"Expected current_state shape {(ACTION_DIM,)}, got {current_state.shape}")
    if raw_actions.ndim != 2 or raw_actions.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected raw_actions shape (T, {ACTION_DIM}), got {raw_actions.shape}")

    clipped_actions = np.empty_like(raw_actions, dtype=np.float32)
    prev = current_state.astype(np.float32).copy()

    for idx, target in enumerate(raw_actions):
        scaled = prev + (target.astype(np.float32) - prev) * float(action_scale)
        delta = scaled - prev
        delta[:6] = np.clip(delta[:6], -max_joint_delta, max_joint_delta)
        delta[6] = np.clip(delta[6], -max_gripper_delta, max_gripper_delta)
        delta[7:13] = np.clip(delta[7:13], -max_joint_delta, max_joint_delta)
        delta[13] = np.clip(delta[13], -max_gripper_delta, max_gripper_delta)
        prev = prev + delta
        clipped_actions[idx] = prev

    return clipped_actions


def summarize_run(stats: dict[str, list[float]] | dict[str, int], *, shadow: bool, chunks: int) -> dict[str, Any]:
    infer_times = np.asarray(stats["infer_ms"], dtype=np.float32) if stats["infer_ms"] else np.asarray([], dtype=np.float32)
    post_times = np.asarray(stats["post_ms"], dtype=np.float32) if stats["post_ms"] else np.asarray([], dtype=np.float32)
    clip_magnitudes = np.asarray(stats["clip_max_abs"], dtype=np.float32) if stats["clip_max_abs"] else np.asarray([], dtype=np.float32)

    def _metric(values: np.ndarray, fn) -> float | None:
        if values.size == 0:
            return None
        return float(fn(values))

    return {
        "shadow": shadow,
        "chunks": chunks,
        "empty_polls": int(stats["empty_polls"]),
        "post_success": int(stats["post_success"]),
        "infer_ms_mean": _metric(infer_times, np.mean),
        "infer_ms_p95": _metric(infer_times, lambda x: np.quantile(x, 0.95)),
        "post_ms_mean": _metric(post_times, np.mean),
        "post_ms_p95": _metric(post_times, lambda x: np.quantile(x, 0.95)),
        "clip_max_abs_mean": _metric(clip_magnitudes, np.mean),
        "clip_max_abs_max": _metric(clip_magnitudes, np.max),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    run_name = args.run_name or f"dexmal_eval_{time.strftime('%Y%m%d_%H%M%S')}"
    jsonl_logger = JsonlLogger(Path(args.log_dir), run_name)
    gateway = RobotGatewayClient(args.robot_host, args.robot_port, args.request_timeout)
    policy = create_policy(args)
    logging.info("Loaded policy. metadata=%s", getattr(policy, "metadata", {}))

    if args.action_horizon != ACTION_HORIZON:
        raise ValueError(f"This eval bridge requires action_horizon={ACTION_HORIZON}")

    stats: dict[str, Any] = {
        "infer_ms": [],
        "post_ms": [],
        "clip_max_abs": [],
        "empty_polls": 0,
        "post_success": 0,
    }

    if args.start_record_on_start:
        logging.info("Sending start-record mode to robot gateway")
        gateway.set_mode(START_RECORD_MODE)
        time.sleep(args.start_record_wait_seconds)

    if args.set_model_mode_on_start:
        logging.info("Switching robot gateway to model mode")
        gateway.set_mode(MODEL_MODE)

    chunk_idx = 0
    empty_poll_log_at = 0.0

    try:
        while args.max_chunks <= 0 or chunk_idx < args.max_chunks:
            loop_start = time.time()
            payload = gateway.get_infer_state()
            if payload is None:
                stats["empty_polls"] += 1
                if time.time() - empty_poll_log_at > 5.0:
                    try:
                        logging.info("Waiting for observation. current_mode=%s", gateway.get_mode())
                    except Exception:
                        logging.warning("Waiting for observation. failed to query current mode.", exc_info=True)
                    empty_poll_log_at = time.time()
                sleep_time = max(0.0, (1.0 / args.poll_hz) - (time.time() - loop_start))
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue

            obs_id, current_state, observation = build_observation(payload, args.prompt)
            infer_start = time.time()
            result = policy.infer(observation)
            infer_ms = 1000.0 * (time.time() - infer_start)
            stats["infer_ms"].append(infer_ms)

            raw_actions = np.asarray(result["actions"], dtype=np.float32)
            if raw_actions.ndim != 2 or raw_actions.shape[1] != ACTION_DIM:
                raise RuntimeError(f"Policy returned invalid action shape {raw_actions.shape}")
            if raw_actions.shape[0] < args.action_horizon:
                raise RuntimeError(
                    f"Policy returned {raw_actions.shape[0]} steps, expected at least {args.action_horizon}"
                )
            if not np.isfinite(raw_actions).all():
                raise RuntimeError("Policy returned NaN/Inf actions.")

            raw_actions = raw_actions[: args.action_horizon]
            clipped_actions = clip_action_chunk(
                current_state,
                raw_actions,
                action_scale=args.action_scale,
                max_joint_delta=args.max_joint_delta,
                max_gripper_delta=args.max_gripper_delta,
            )

            clip_max_abs = float(np.max(np.abs(clipped_actions - raw_actions)))
            stats["clip_max_abs"].append(clip_max_abs)

            post_start = time.time()
            post_result = gateway.post_action_chunk(
                obs_id=obs_id,
                actions=clipped_actions,
                dry_run=args.shadow,
                action_source="dexmal_eval_bridge",
            )
            post_ms = 1000.0 * (time.time() - post_start)
            stats["post_ms"].append(post_ms)
            stats["post_success"] += 1

            record = {
                "chunk_idx": chunk_idx,
                "obs_id": obs_id,
                "shadow": args.shadow,
                "infer_ms": infer_ms,
                "post_ms": post_ms,
                "clip_max_abs": clip_max_abs,
                "raw_first_action": np.round(raw_actions[0], 6).tolist(),
                "clipped_first_action": np.round(clipped_actions[0], 6).tolist(),
                "raw_last_action": np.round(raw_actions[-1], 6).tolist(),
                "clipped_last_action": np.round(clipped_actions[-1], 6).tolist(),
                "post_result": post_result,
                "policy_timing": result.get("policy_timing", {}),
            }
            jsonl_logger.write_chunk(record)

            logging.info(
                "chunk=%d obs_id=%s infer_ms=%.1f post_ms=%.1f shadow=%s clip_max_abs=%.5f",
                chunk_idx,
                obs_id,
                infer_ms,
                post_ms,
                args.shadow,
                clip_max_abs,
            )
            chunk_idx += 1

            sleep_time = max(0.0, (1.0 / args.poll_hz) - (time.time() - loop_start))
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        if args.set_stop_mode_on_exit:
            try:
                logging.info("Switching robot gateway to stop mode")
                gateway.set_mode(STOP_MODE)
            except Exception:
                logging.exception("Failed to switch robot gateway to stop mode on exit")

        summary = summarize_run(stats, shadow=args.shadow, chunks=chunk_idx)
        jsonl_logger.write_summary(summary)
        logging.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
