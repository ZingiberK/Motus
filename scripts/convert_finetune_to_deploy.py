#!/usr/bin/env python3
"""Convert a Phase-1 training checkpoint into a deploy-loadable directory.

The native trainer (accelerate + DeepSpeed zero1) writes::

    checkpoint_step_<N>/
        pytorch_model/mp_rank_00_model_states.pt   # {'module': state_dict, ...}
        pytorch_model_0.bin                        # raw model.state_dict()
        config.json

Deploy (``inference/robotwin/Motus/deploy_policy.MotusPolicy.load_checkpoint``)
expects a directory that directly contains::

    mp_rank_00_model_states.pt   # {'module': state_dict}

This tool extracts the model state dict from a training checkpoint and writes it
in the deploy layout so ``--motus_ckpt <out_dir>`` works for eval and RL.

    python scripts/convert_finetune_to_deploy.py \\
        --in checkpoints_motus_finetune/checkpoint_step_50000 \\
        --out deploy_ckpts/motus_finetune_50000
"""

import argparse
import json
import shutil
from pathlib import Path

import torch


def extract_state_dict(ckpt_dir: Path) -> dict:
    ds = ckpt_dir / "pytorch_model" / "mp_rank_00_model_states.pt"
    if ds.exists():
        blob = torch.load(ds, map_location="cpu")
        return blob["module"] if isinstance(blob, dict) and "module" in blob else blob
    bin0 = ckpt_dir / "pytorch_model_0.bin"
    if bin0.exists():
        return torch.load(bin0, map_location="cpu")
    flat = ckpt_dir / "mp_rank_00_model_states.pt"
    if flat.exists():
        blob = torch.load(flat, map_location="cpu")
        return blob["module"] if isinstance(blob, dict) and "module" in blob else blob
    raise FileNotFoundError(
        f"No model weights found under {ckpt_dir} "
        f"(looked for pytorch_model/mp_rank_00_model_states.pt, pytorch_model_0.bin)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="training checkpoint_step_<N> dir")
    ap.add_argument("--out", required=True, help="output deploy dir")
    args = ap.parse_args()

    src = Path(args.inp)
    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)

    state_dict = extract_state_dict(src)
    out_file = dst / "mp_rank_00_model_states.pt"
    torch.save({"module": state_dict}, out_file)
    print(f"Wrote {out_file} ({len(state_dict)} tensors)")

    cfg = src / "config.json"
    if cfg.exists():
        shutil.copy2(cfg, dst / "config.json")
        print(f"Copied config.json -> {dst / 'config.json'}")


if __name__ == "__main__":
    main()
