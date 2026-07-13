"""Offline smoke test for Motus SDEdit action refinement.

Validates the mechanism (ckpt loads, forward runs, t0 interpolation) WITHOUT RoboTwin:
feeds a real (first_frame, state, a_vla, instruction) tuple pulled from an existing
vla_zero wm_trace and checks that:
  - start_t -> 1.0  => output ~ pure Motus action (far from a_vla)
  - start_t small   => output ~ a_vla (near a_vla, Motus barely edits)

NOTE: uses head-only first_frame (composite multi-cam obs only matters for closed-loop SR);
this test only checks the SDEdit denoising mechanism / action-space assumption (raw qpos).

Usage (motus env):
  python sdedit_offline_smoke.py \
    --ckpt /mnt/data14/liuxiao/pretrained_models/Motus_robotwin2 \
    --wan  /mnt/data14/liuxiao/pretrained_models/Wan2.2-TI2V-5B \
    --vlm  /mnt/data14/liuxiao/pretrained_models/Qwen3-VL-2B-Instruct \
    --wm_trace <path-to-a-vla_zero-wm_trace.pt>
"""
import argparse
import glob
from pathlib import Path

import torch


def pick_wm_trace(explicit: str) -> str:
    if explicit:
        return explicit
    root = "/mnt/data14/yyg/wrm_rl_runs/ppo_v3"
    cands = glob.glob(f"{root}/*/round*/wm_traces/*_ep1_chunk0.pt")
    if not cands:
        cands = glob.glob(f"{root}/*/round*/wm_traces/*.pt")
    assert cands, "no vla_zero wm_trace found"
    return sorted(cands)[0]


def rms(x: torch.Tensor) -> float:
    return float((x.float() ** 2).mean().sqrt())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/data14/liuxiao/pretrained_models/Motus_robotwin2")
    ap.add_argument("--wan", default="/mnt/data14/liuxiao/pretrained_models/Wan2.2-TI2V-5B")
    ap.add_argument("--vlm", default="/mnt/data14/liuxiao/pretrained_models/Qwen3-VL-2B-Instruct")
    ap.add_argument("--wm_trace", default="")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--t0", default="0.2,0.4,0.6,0.8,1.0")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    import sys
    sys.path.append(str(here))
    from deploy_policy import MotusPolicy
    from PIL import Image

    dev = "cuda"
    print(f"[load] Motus ckpt={args.ckpt}")
    policy = MotusPolicy(
        checkpoint_path=args.ckpt, config_path=str(here / "utils" / "robotwin.yml"),
        wan_path=args.wan, vlm_path=args.vlm, device=dev, task_name="sdedit_smoke",
    )
    policy.save_images = False
    model = policy.model

    wm_path = pick_wm_trace(args.wm_trace)
    print(f"[data] wm_trace={wm_path}")
    tr = torch.load(wm_path, map_location="cpu")
    first_frame = tr["first_frame"].float()          # [3,384,320] in [0,1]
    if first_frame.max() > 1.5:
        first_frame = first_frame / 255.0
    state = tr["state"].float()                       # [14]
    a_vla = tr["a_vla_real"].float()                  # [50,14] raw qpos
    instruction = tr["meta"]["instruction"]
    print(f"[data] a_vla shape={tuple(a_vla.shape)} state={tuple(state.shape)} instr='{instruction[:60]}...'")

    scene_prefix = ("The whole scene is in a realistic, industrial art style with three views: "
                    "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
                    "The aloha robot is currently performing the following task: ")
    full_instr = f"{scene_prefix}{instruction}"
    t5_out = policy.t5_encoder([full_instr], dev)
    t5_list = [t5_out.squeeze(0)] if isinstance(t5_out, torch.Tensor) and t5_out.dim() == 3 else (
        t5_out if isinstance(t5_out, list) else [t5_out])
    pil = Image.fromarray((first_frame.permute(1, 2, 0).numpy() * 255).astype("uint8"), "RGB")
    vlm_inputs = policy._preprocess_vlm_messages(full_instr, pil)

    ff = first_frame.unsqueeze(0).to(dev)             # [1,3,384,320]
    st = state.unsqueeze(0).to(dev)                   # [1,14]
    chunk = model.config.action_chunk_size
    ds = model.config.global_downsample_rate          # 3
    print(f"[cfg] motus action_chunk_size={chunk} global_downsample_rate={ds} steps={args.steps}")

    # frequency alignment: Motus samples actions at raw offsets [ds,2ds,...,chunk*ds]
    # (0-based idx ds-1, 2ds-1, ...). Our VLA is spacing-1 (50 consecutive). Resample.
    idx_stride = [min((i + 1) * ds - 1, a_vla.shape[0] - 1) for i in range(chunk)]  # [2,5,...,47]
    a_init_stride = a_vla[idx_stride]                 # [16,14] Motus-grid aligned
    a_init_first = a_vla[:chunk]                      # [16,14] naive first-16 (misaligned)

    # pure Motus
    with torch.no_grad():
        _, a_motus = model.inference_step(ff, st, num_inference_steps=args.steps,
                                          language_embeddings=t5_list, vlm_inputs=[vlm_inputs])
    a_motus = a_motus.squeeze(0).cpu()               # [16,14]

    # --- frequency alignment check: which resampling of a_vla matches Motus's native grid? ---
    print(f"\n[freq-align] rms(a_motus - a_vla_stride3)={rms(a_motus - a_init_stride):.4f}  "
          f"rms(a_motus - a_vla_first16)={rms(a_motus - a_init_first):.4f}  (smaller = better aligned)")
    # endpoint reach from current state (how far each traj goes)
    print(f"[reach] |a_vla[-1]-state|={rms(a_vla[-1]-state):.4f}  |a_vla_stride3[-1]-state|={rms(a_init_stride[-1]-state):.4f}  "
          f"|a_motus[-1]-state|={rms(a_motus[-1]-state):.4f}")

    a_init = a_init_stride.unsqueeze(0).to(dev)       # use freq-aligned init
    print("\nstart_t | rms(ref-a_vla_aligned) | rms(ref-a_motus)   (small t0 -> near VLA, t0=1 -> near motus)")
    for t0 in [float(x) for x in args.t0.split(",")]:
        _, ref = model.sdedit_inference_step(
            ff, st, action_init=a_init, start_t=t0, num_inference_steps=args.steps,
            language_embeddings=t5_list, vlm_inputs=[vlm_inputs], seed=0)
        ref = ref.squeeze(0).cpu()
        print(f"  {t0:.2f}   |   {rms(ref - a_init_stride):.4f}      |    {rms(ref - a_motus):.4f}")


if __name__ == "__main__":
    main()
