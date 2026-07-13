# Motus RL：Flow-SDE PPO + 世界模型监督

> 从 `WRM_RL_PLAN_PPO.md` **适配到 Motus**。配方不变：**视频分支走监督 FM，动作分支走 PPO（Flow-SDE）**，梯度隔离。  
> 实现：`motus_rl/`；原 WRM learner 保留在 `legacy/wrm_rl/` 作对照，避免新机重写偏题。

---

## 0. 与 WRM 版的关系

| | WRM（legacy） | Motus（本方案） |
|--|--|--|
| 策略对象 | residual `Δ`，`a = a_vla + Δ` | **绝对动作** raw qpos（可 SDEdit 锚定 `a_vla`） |
| 模型 | `WRMUnd` | `Motus`（WAN + action expert + und / Qwen3-VL） |
| SDE API | `sample_actions_sde` / `action_logprob_from_trace` | 同名方法，已加在 `inference/.../models/motus.py` |
| 视频监督 | `video_supervised_loss(first_frame, futures, …, delta_exec)` | `video_supervised_loss(…, actions_exec)` → 复用 `training_step` 视频管线 |
| Learner | `legacy/wrm_rl/ppo_update.py` | `motus_rl/ppo_update.py` |
| 观测 | 单视角 head + und_feats | **三视角 composite** + 在线 T5/VLM |

**不要改的核心**（πRL / RLinf 配方）：

1. WAN **只吃** `L_video` 监督，**不吃**策略梯度（两次 WRM 崩盘根因）。  
2. 动作 Flow-SDE：`σ = η√|dt|`，`logp = Σ log N(x_{i+1}; x_i+v·dt, σ²)`，PPO-clip。  
3. 优势 = 折扣 success − **任务均值**（默认关冷 critic，`c_value=0`）。  
4. `η=0.5`，`lr/wan_lr=1e-6`，`target_kl=0.5` epoch 早停。

---

## 1. 目标隔离

| 分支 | 目标 | 信号 | requires_grad |
|------|------|------|----------------|
| Video (WAN) | 预测「执行动作后的真实未来」（含失败） | 监督 FM | video pass: WAN+und |
| Action | 最大化 success | PPO Flow-SDE | action pass: action+und |
| Und | 共享语义 | 两边都回传 | 两 pass 都开 |

```
L_total ≈ L_action_PPO + β · L_video_supervised     # β=0.5
```

---

## 2. Motus Flow-SDE

已实现（`Motus.sample_actions_sde`）：

- 从噪声（或 SDEdit `action_init` + `start_t`）出发，Euler–Maruyama 去噪。  
- 返回 `action`, `action_latents[K+1]`, `video_latents[K+1]`, `timesteps`, `eta`, `old_logprob`。  
- `action_logprob_from_trace` 在 θ 下重算 logp → `ratio = exp(logp_new − logp_old)`。

**SDEdit + RL**：`start_t<1` + `action_init=a_vla_stride3` 可在 VLA 邻域探索；`start_t=1` = 纯 Motus SDE。

---

## 3. 视频如何见到错误

闭环执行 Motus（SDE）动作 → 仿真渲染真实未来（大量失败）→  
`L_video = FM(WAN预测, encode(观测未来))`，动作条件为 **实际执行的 absolute qpos**。  
语义与 Motus `training_step` 一致，只是目标从 demo 成功未来变成 RL 含失败未来。

注意：Motus 视频要 **composite 三视角**；rollout 需存 composite `first_frame` 或三路 cam + 与 SFT 对齐的 future 窗口。

---

## 4. PPO 细节（同 WRM 文档 §3）

- 环境级：chunk = 一步；`return_t = γ^(T-1-t)·success`；`A = return − mean_task(return)`。  
- 可选 `--adv_mode paired`：配对 VLA/ref episode，`A ∝ (success_motus − success_ref)`。  
- `clip_eps=0.2`，`epochs≥2` + `target_kl=0.5`。  
- 纯 success，无 `−λ‖a‖`。

---

## 5. 数据采集（待接 rollout；schema 已定）

每个 chunk 存 `.pt`（见 `motus_rl/ppo_update.py` docstring）：

- `action_latents / video_latents / timesteps / eta / old_logprob`  
- `state`, `action`（执行动作）, `first_frame`（composite [0,1]）或三 cam  
- `instruction`；可选缓存 `und_tokens`  
- episode json：`trace_paths`, `future_paths`, `success`, `mode`（`motus` / `vla_ref`）

**实现状态**：模型 API + learner 已就绪；**Motus SDE rollout server**（替代 WRM `lingbot_wrm_rl_server` 的 SDE 路径）需在新机接：`sample_actions_sde` → 存 trace → worker 执行 `action`。可复用现有 `rl_rollout_worker` 骨架，把 Motus refine 的 `infer` 换成返回 SDE trace 的接口。

---

## 6. Learner 用法

```bash
export MOTUS_INFER_ROOT=$HELIX/Motus/inference/robotwin/Motus
cd $HELIX/Motus
conda activate motus
python -m motus_rl.ppo_update \
  --tasks stack_blocks_three handover_block \
  --rl_root $HELIX/wrm_rl_runs/motus_ppo \
  --round 0 \
  --motus_ckpt $HELIX/pretrained/Motus_robotwin2 \
  --wan $HELIX/pretrained/Wan2.2-TI2V-5B \
  --vlm $HELIX/pretrained/Qwen3-VL-2B-Instruct \
  --out_ckpt $HELIX/wrm_rl_runs/motus_ppo/ckpts/round1.pt \
  --lr 1e-6 --wan_lr 1e-6 --eta 0.5 --epochs 2 --target_kl 0.5 \
  --group_adv 1 --video_beta 0.5
# 产出 round1.pt + round1_action_expert.pt（可喂 motus_sdedit_server --ft_action_ckpt）
```

关键超参表同 `WRM_RL_PLAN_PPO.md` §7（η / clip / lr / β / group_adv / target_kl）。

---

## 7. 文件清单

| 路径 | 作用 |
|------|------|
| `inference/.../models/motus.py` | `sample_actions_sde`, `action_logprob_from_trace`, `video_supervised_loss`, `_joint_velocities` |
| `motus_rl/reward.py` | 折扣回报 / paired baseline |
| `motus_rl/ppo_update.py` | Motus PPO learner |
| `legacy/wrm_rl/*` | 原 WRM 实现（对照，勿当 Motus 入口） |
| `MOTUS_RL_PLAN_PPO.md` | 本文 |

---

## 8. 成功判据

- held-out eval macro SR 相对 Motus-FT / 纯 Motus / VLA 有净提升，且饱和任务不塌。  
- `LOGP_CHECK`：epoch0 重算 logp ≈ stored `old_logprob`（ratio≈1）。  
- WM loss 在含失败 futures 上下降。  
- PPO `ratio ∈ [0.8, 1.2]`，`target_kl` 能刹停漂移。
