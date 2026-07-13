"""Tier 1: BC-finetune Motus's ACTION expert toward VLA (stride-3 raw qpos).

Manifold-gap probe (see WRM_RL_PLAN_MOTUS_REFINE.md §10). Trains ONLY the action
expert (WAN video model, Qwen3-VL und, T5, video expert all frozen) with a
flow-matching loss toward the collected VLA targets, under Motus's own obs
conditioning (composite 3-view first frame + raw-qpos state + language). After
finetuning we re-eval SDEdit low-t0 refine on HELD-OUT eval seeds: if closing the
manifold gap revives the blend -> manifold confirmed; if not -> deeper misalignment.

Data: collect_motus_ft.sh output
    <data_root>/<task>/round0/motus_ft/<task>_seed<seed>_ep<ep>.npz
      cam_high/left/right [n_chunk,H,W,3] uint8 | state [n_chunk,14] | target [n_chunk,16,14] | instruction

Run (motus env):
    python finetune_motus_action.py \
        --data_root /mnt/data14/yyg/wrm_rl_runs/motus_ft \
        --out /mnt/data14/yyg/Motus/runs/motus_ft_action_v1 \
        --epochs 3 --lr 1e-4 --accum 16
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

MOTUS_INFER = os.environ.get("MOTUS_INFER_ROOT", "/mnt/data14/yyg/Motus/inference/robotwin/Motus")
sys.path.insert(0, MOTUS_INFER)

from deploy_policy import MotusPolicy  # noqa: E402


def build_file_index(data_root: str, tasks: set | None) -> list[str]:
    files = []
    for npz in sorted(glob.glob(os.path.join(data_root, "*", "round0", "motus_ft", "*.npz"))):
        task = Path(npz).parts[-4]                       # <data_root>/<task>/round0/motus_ft/<f>
        if tasks and task not in tasks:
            continue
        files.append(npz)
    return files


def prep_inputs(policy: MotusPolicy, cam_high, cam_left, cam_right, state, instruction):
    """Build Motus conditioning exactly like MotusPolicy.get_action (composite frame,
    T5 embeddings, VLM inputs). Returns (first_frame[1,3,H,W], state[1,14], t5_list, vlm_inputs)."""
    obs = {
        "observation": {
            "head_camera": {"rgb": np.asarray(cam_high)},
            "left_camera": {"rgb": np.asarray(cam_left)},
            "right_camera": {"rgb": np.asarray(cam_right)},
        },
        "joint_action": {"vector": np.asarray(state, dtype=np.float32)},
    }
    policy.set_instruction(instruction)
    policy.update_obs(obs)                               # identical resize_with_padding pipeline
    current_frame = policy.obs_cache[-1]                 # [1,3,384,320] in [0,1]
    current_state = policy.current_state                 # [1,14] raw qpos

    scene_prefix = ("The whole scene is in a realistic, industrial art style with three views: "
                    "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
                    "The aloha robot is currently performing the following task: ")
    full_instr = f"{scene_prefix}{instruction}"
    t5_out = policy.t5_encoder([full_instr], policy.device)
    if isinstance(t5_out, torch.Tensor):
        t5_list = [t5_out.squeeze(0)] if t5_out.dim() == 3 else [t5_out]
    else:
        t5_list = t5_out
    first_frame_pil = policy._tensor_to_pil_image(current_frame.squeeze(0).cpu())
    vlm_inputs = policy._preprocess_vlm_messages(full_instr, first_frame_pil)
    return current_frame, current_state, t5_list, vlm_inputs


def run_file(policy, npz_path, chunk_order, timestep_sample, sigmoid_scale, train: bool):
    """Yield per-chunk FM loss for one episode npz (chunks visited in chunk_order)."""
    try:
        d = np.load(npz_path, allow_pickle=True)
        instr = str(d["instruction"])
        _ = d["cam_high"].shape  # touch to force-read / validate archive
    except Exception as e:  # skip corrupt archive without killing the run
        print(f"[warn] skip unreadable npz {npz_path}: {e}", flush=True)
        return
    ch, cl, cr = d["cam_high"], d["cam_left"], d["cam_right"]
    st, tg = d["state"], d["target"]
    for c in chunk_order:
        ff, state, t5_list, vlm_inputs = prep_inputs(policy, ch[c], cl[c], cr[c], st[c], instr)
        target = torch.from_numpy(np.asarray(tg[c], dtype=np.float32)).unsqueeze(0)  # [1,16,14]
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            loss = policy.model.action_fm_loss(
                first_frame=ff, state=state, action_target=target,
                language_embeddings=t5_list, vlm_inputs=[vlm_inputs],
                timestep_sample=timestep_sample, sigmoid_scale=sigmoid_scale,
            )
        yield loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/mnt/data14/yyg/wrm_rl_runs/motus_ft")
    ap.add_argument("--tasks", default="", help="comma-separated subset; empty = all collected tasks")
    ap.add_argument("--motus_ckpt", default="/mnt/data14/liuxiao/pretrained_models/Motus_robotwin2")
    ap.add_argument("--wan", default="/mnt/data14/liuxiao/pretrained_models/Wan2.2-TI2V-5B")
    ap.add_argument("--vlm", default="/mnt/data14/liuxiao/pretrained_models/Qwen3-VL-2B-Instruct")
    ap.add_argument("--out", default="/mnt/data14/yyg/Motus/runs/motus_ft_action_v1")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=16, help="grad-accum micro-steps (B=1 each)")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--timestep_sample", default="logit_normal", choices=["logit_normal", "uniform"])
    ap.add_argument("--sigmoid_scale", type=float, default=1.0)
    ap.add_argument("--save_every_steps", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--max_files", type=int, default=0, help="cap #episodes (debug); 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.log"

    def log(msg):
        line = f"[{time.strftime('%F %T')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    tasks = set(t for t in args.tasks.split(",") if t) or None
    files = build_file_index(args.data_root, tasks)
    if args.max_files > 0:
        files = files[:args.max_files]
    if not files:
        raise RuntimeError(f"no motus_ft npz under {args.data_root} (tasks={tasks})")
    rng = random.Random(args.seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * args.val_frac))
    val_files, train_files = files[:n_val], files[n_val:]
    log(f"files: total={len(files)} train={len(train_files)} val={len(val_files)} tasks={'all' if not tasks else len(tasks)}")

    cfg = str(Path(MOTUS_INFER) / "utils" / "robotwin.yml")
    policy = MotusPolicy(checkpoint_path=args.motus_ckpt, config_path=cfg,
                         wan_path=args.wan, vlm_path=args.vlm, device="cuda", task_name="motus_ft")
    policy.save_images = False
    model = policy.model
    model.eval()  # frozen backbones deterministic; requires_grad controls what trains

    n_train_p = 0
    for name, p in model.named_parameters():
        p.requires_grad = "action_expert" in name
        if p.requires_grad:
            n_train_p += p.numel()
    log(f"trainable (action_expert) params: {n_train_p/1e6:.2f}M")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))

    @torch.no_grad()
    def validate():
        tot, n = 0.0, 0
        for vf in val_files:
            d = np.load(vf, allow_pickle=True)
            n_chunk = d["state"].shape[0]
            for loss in run_file(policy, vf, list(range(n_chunk)),
                                 args.timestep_sample, args.sigmoid_scale, train=False):
                tot += float(loss); n += 1
        return tot / max(n, 1)

    gstep = 0
    micro = 0
    running = 0.0
    opt.zero_grad(set_to_none=True)
    t_start = time.time()
    for ep in range(args.epochs):
        rng.shuffle(train_files)
        for fi, f in enumerate(train_files):
            d = np.load(f, allow_pickle=True)
            n_chunk = int(d["state"].shape[0])
            order = list(range(n_chunk))
            rng.shuffle(order)
            for loss in run_file(policy, f, order, args.timestep_sample, args.sigmoid_scale, train=True):
                (loss / args.accum).backward()
                running += float(loss)
                micro += 1
                if micro % args.accum == 0:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad], args.grad_clip)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    gstep += 1
                    if gstep % args.log_every == 0:
                        rate = micro / (time.time() - t_start)
                        log(f"ep{ep} gstep{gstep} file{fi}/{len(train_files)} "
                            f"loss={running/args.accum/args.log_every:.5f} {rate:.1f}micro/s")
                        running = 0.0
                    if gstep % args.save_every_steps == 0:
                        ckpt = out / f"action_expert_step{gstep:06d}.pt"
                        torch.save(model.action_expert.state_dict(), ckpt)
                        log(f"saved {ckpt}")
        vloss = validate()
        ckpt = out / f"action_expert_ep{ep}.pt"
        torch.save(model.action_expert.state_dict(), ckpt)
        log(f"=== epoch {ep} done | val_loss={vloss:.5f} | saved {ckpt} ===")

    torch.save(model.action_expert.state_dict(), out / "action_expert_final.pt")
    log("training done.")


if __name__ == "__main__":
    main()
