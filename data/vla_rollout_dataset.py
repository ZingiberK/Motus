"""VLA-rollout dataset for Phase 1 Motus finetune.

Reads the per-episode npz produced by the RoboTwin collection worker
(``RoboTwin/script/rl_rollout_worker.py``) and yields Motus-native training
samples for the joint video+action objective (``models/motus.training_step``).

npz layout (one file per successful episode, ``<root>/**/traj/*.npz``)::

    cam_high/left/right    : [n_chunk, H, W, 3] uint8   (chunk first frame, 3 raw views)
    future_high/left/right : [n_chunk, T, H, W, 3] uint8 (future frames, 3 raw views)
    state                  : [n_chunk, 14] float32       (raw qpos)
    target                 : [n_chunk, 16, 14] float32    (action chunk, raw qpos)
    instruction            : scalar str

Each sample composites the 3 raw views into the exact frame Motus consumes at
inference (see ``inference/robotwin/Motus/deploy_policy.MotusPolicy.update_obs``):
head on top, left|right (each resized to 160x120) below, then
``resize_with_padding`` to ``video_size`` and normalize to [0, 1].

Language embeddings (WAN UMT5-xxl) are heavy, so they are pre-encoded once into
``<dataset_dir>/lang_cache.pt`` by ``scripts/build_lang_cache.py`` and looked up
here by raw instruction string.
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.utils.data as data
from transformers import AutoProcessor

from data.utils.image_utils import resize_with_padding
from utils.vlm_utils import preprocess_vlm_messages

logger = logging.getLogger(__name__)

# Must match deploy_policy.MotusPolicy (get_action / get_action_sde) so that
# Phase 1 finetune, Phase 2 RL and deployment all condition on the same prompt.
SCENE_PREFIX = (
    "The whole scene is in a realistic, industrial art style with three views: "
    "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
    "The aloha robot is currently performing the following task: "
)


def build_composite(head: np.ndarray, left: np.ndarray, right: np.ndarray,
                    video_size: Tuple[int, int]) -> torch.Tensor:
    """3 raw views -> composite frame [3, H, W] in [0, 1] (matches update_obs)."""
    left_r = cv2.resize(left, (160, 120))
    right_r = cv2.resize(right, (160, 120))
    bottom = np.concatenate([left_r, right_r], axis=1)
    image = np.concatenate([head, bottom], axis=0)
    image = resize_with_padding(image, video_size)
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).float()


class VLARolloutDataset(data.Dataset):
    def __init__(
        self,
        dataset_dir: str,
        global_downsample_rate: int = 3,
        video_action_freq_ratio: int = 2,
        num_video_frames: int = 8,
        video_size: Tuple[int, int] = (384, 320),
        max_episodes: Optional[int] = None,
        val: bool = False,
        image_aug: bool = False,
        vlm_checkpoint_path: Optional[str] = None,
        lang_cache_path: Optional[str] = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.global_downsample_rate = global_downsample_rate
        self.video_action_freq_ratio = video_action_freq_ratio
        self.num_video_frames = num_video_frames
        self.action_chunk_size = num_video_frames * video_action_freq_ratio
        self.video_size = tuple(video_size)
        self.val = val
        self.image_aug = image_aug

        # ---- discover episodes + build a flat (file, chunk) index ----
        self.npz_files: List[Path] = sorted(self.dataset_dir.glob("**/traj/*.npz"))
        if max_episodes is not None:
            self.npz_files = self.npz_files[:max_episodes]
        if not self.npz_files:
            raise ValueError(f"No traj/*.npz found under {self.dataset_dir}")

        self.index: List[Tuple[int, int]] = []
        for fi, fp in enumerate(self.npz_files):
            try:
                with np.load(fp) as d:
                    n_chunk = int(d["state"].shape[0])
            except Exception as e:
                logger.warning(f"skip unreadable npz {fp}: {e}")
                continue
            self.index.extend((fi, ci) for ci in range(n_chunk))
        if not self.index:
            raise ValueError(f"No usable chunks under {self.dataset_dir}")
        logger.info(
            f"VLARolloutDataset: {len(self.npz_files)} episodes, {len(self.index)} chunks, "
            f"video_size={self.video_size}, chunk={self.action_chunk_size}, T={num_video_frames}"
        )

        # ---- language embedding cache (pre-encoded UMT5) ----
        cache_path = Path(lang_cache_path) if lang_cache_path else self.dataset_dir / "lang_cache.pt"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Language cache not found: {cache_path}. Build it first with "
                f"`python scripts/build_lang_cache.py --dataset_dir {self.dataset_dir} --wan <WAN_PATH>`."
            )
        self.lang_cache: Dict[str, torch.Tensor] = torch.load(cache_path, map_location="cpu")
        logger.info(f"Loaded {len(self.lang_cache)} language embeddings from {cache_path}")

        # ---- VLM processor (tokenizer + image processor) ----
        self.vlm_processor = None
        if vlm_checkpoint_path is not None:
            self.vlm_processor = AutoProcessor.from_pretrained(vlm_checkpoint_path, trust_remote_code=True)
            logger.info(f"VLM processor loaded from {vlm_checkpoint_path}")
        else:
            logger.warning("vlm_checkpoint_path not provided; vlm_inputs will be None")

    def __len__(self) -> int:
        return len(self.index)

    def _sample(self, fi: int, ci: int) -> Dict[str, Any]:
        fp = self.npz_files[fi]
        with np.load(fp) as d:
            instruction = str(d["instruction"])
            head = d["cam_high"][ci]
            left = d["cam_left"][ci]
            right = d["cam_right"][ci]
            fh = d["future_high"][ci]
            fl = d["future_left"][ci]
            frr = d["future_right"][ci]
            state = torch.from_numpy(np.asarray(d["state"][ci], dtype=np.float32))
            action_sequence = torch.from_numpy(np.asarray(d["target"][ci], dtype=np.float32))

        first_frame = build_composite(head, left, right, self.video_size)  # [3,H,W]
        T = self.num_video_frames
        video_frames = torch.stack([
            build_composite(fh[t], fl[t], frr[t], self.video_size) for t in range(T)
        ])  # [T,3,H,W]

        if instruction not in self.lang_cache:
            raise KeyError(f"instruction not in lang cache: {instruction!r} (rebuild lang_cache.pt)")
        language_embedding = self.lang_cache[instruction].float()

        vlm_inputs = None
        if self.vlm_processor is not None:
            pil = _tensor_to_pil(first_frame)
            vlm_inputs = preprocess_vlm_messages(f"{SCENE_PREFIX}{instruction}", pil, self.vlm_processor)

        return {
            "first_frame": first_frame,
            "video_frames": video_frames,
            "initial_state": state,
            "action_sequence": action_sequence,
            "language_embedding": language_embedding,
            "vlm_inputs": vlm_inputs,
        }

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        order = [idx % len(self.index)] + random.sample(range(len(self.index)), k=min(7, len(self.index)))
        for j in order:
            fi, ci = self.index[j]
            try:
                return self._sample(fi, ci)
            except Exception as e:
                logger.warning(f"retry sample ({self.npz_files[fi].name}[{ci}]): {e}")
                continue
        return None


def _tensor_to_pil(tensor_chw: torch.Tensor):
    from PIL import Image
    arr = tensor_chw.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")
