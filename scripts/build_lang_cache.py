#!/usr/bin/env python3
"""Pre-encode VLA-rollout instructions into a WAN UMT5 embedding cache.

Scans ``<dataset_dir>/**/traj/*.npz`` for unique instruction strings and encodes
``SCENE_PREFIX + instruction`` with the WAN UMT5-xxl text encoder (the same one
used at deploy time), writing ``{raw_instruction: [seq, dim] tensor}`` to
``<dataset_dir>/lang_cache.pt``.

Run in the Motus training env (``wan`` importable), on a GPU:

    cd Motus
    python scripts/build_lang_cache.py \\
        --dataset_dir /path/to/motus_v1 \\
        --wan /path/to/Wan2.2-TI2V-5B
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.vla_rollout_dataset import SCENE_PREFIX
from wan.modules.t5 import T5EncoderModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--wan", required=True, help="WAN dir with models_t5_umt5-xxl-enc-bf16.pth + google/umt5-xxl")
    ap.add_argument("--out", default=None, help="default <dataset_dir>/lang_cache.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--text_len", type=int, default=512)
    args = ap.parse_args()

    root = Path(args.dataset_dir)
    out = Path(args.out) if args.out else root / "lang_cache.pt"

    instrs = set()
    for fp in sorted(root.glob("**/traj/*.npz")):
        try:
            with np.load(fp) as d:
                instrs.add(str(d["instruction"]))
        except Exception as e:
            print(f"skip {fp}: {e}")
    instrs = sorted(instrs)
    if not instrs:
        raise SystemExit(f"No instructions found under {root}")
    print(f"Found {len(instrs)} unique instructions")

    encoder = T5EncoderModel(
        text_len=args.text_len,
        dtype=torch.bfloat16,
        device=args.device,
        checkpoint_path=str(Path(args.wan) / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(Path(args.wan) / "google" / "umt5-xxl"),
    )

    cache = {}
    for i, raw in enumerate(instrs):
        out_t = encoder([f"{SCENE_PREFIX}{raw}"], args.device)
        if isinstance(out_t, (list, tuple)):
            emb = out_t[0]
        elif torch.is_tensor(out_t) and out_t.dim() == 3:
            emb = out_t.squeeze(0)
        else:
            emb = out_t
        cache[raw] = emb.detach().float().cpu()
        if (i + 1) % 20 == 0 or i + 1 == len(instrs):
            print(f"  encoded {i + 1}/{len(instrs)}")

    torch.save(cache, out)
    print(f"Saved {len(cache)} embeddings -> {out}")


if __name__ == "__main__":
    main()
