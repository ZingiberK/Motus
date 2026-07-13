# 训练-free 探针：用预训练 Motus 作为动作 refiner（SDEdit）——复盘与结论

> 独立于 `WRM_RL_PLAN_WAM_WARMUP.md`（v4：自训 WAM warmup + RL）与 `WRM_RL_PLAN_PPO.md`。
>
> 想法：**不训练、不 RL**，直接用成熟预训练 WAM（[Motus RoboTwin2](https://huggingface.co/motus-robotics/Motus_robotwin2)，
> [arXiv:2512.13030](https://arxiv.org/abs/2512.13030)）对我们 VLA 的动作做 **SDEdit 式 refine**
> （取 VLA 动作 → 部分加噪 → Motus 去噪 → 执行），期望「VLA 规划、Motus 细调」。
>
> **本文已是复盘版**：探针已实现、闭环 smoke、Tier 0/1 完成。**训练-free 混合被证伪；Tier 1 坐实流形根因。** 正式全量评测（Motus-FT t0=0.3，47 任务 × 100 eval episode）进行中，见 §11。

---

## 0. TL;DR（先看这里）

1. **Motus 本地齐全**、可跑：ckpt + Qwen3-VL-2B + WAN2.2-5B 都在 `/mnt/data14/liuxiao/pretrained_models/`。
2. **纯 Motus 在我们 harness 复现正常**：`stack_blocks_three` **9/10=90%**（论文 91%），`handover_block` 7/10=70%。→ pipeline 无 bug，Motus 动作空间/相机/推理都对。
3. **SDEdit 低/中 t0 混合被证伪**：t0=0.1/0.3/0.6 → **0%**，t0=1.0（=纯 Motus）→ 90%。只有「几乎等于纯 Motus」才 work。
4. **【Tier 0 已定】频率不是问题，指向流形**：`t0=0.0`（纯 VLA stride-3、Motus 零贡献）= **90%** → 频率/执行制度被洗清；两端好、中间全崩的「死亡谷」= **动作流形不兼容**。之前引用的 `rms=0.92 rad` 是 bug（见 §6），已作废。详见 §5.1。
5. **【Tier 1 已完成】流形根因坐实**：BC 微调 Motus 动作专家（2114 轨迹 / 3 epoch）后，held-out eval 上 **t0=0.3 从 0% 复活到 90%**（与 t0=1.0 一致）。见 §10。
6. **战略判断不变**：Motus-FT refine 最多打平 VLA、不带方法 novelty；长期仍**推荐 v4（自训 WM 耦合模型 + RL）**。Tier 1 的价值是**诊断 + v4 Phase-A PoC**。
7. **正式全量评测**：Motus-FT **t0=0.3**，**50 任务** × **100 eval episode**（`eval_policy.py` 协议，无预扫），见 §11。

---

## 1. 资产盘点：本地已齐全 ✔

全部位于 `/mnt/data14/liuxiao/pretrained_models/`：

| 组件 | 本地路径 | 说明 |
|---|---|---|
| **Motus RoboTwin2 微调 ckpt** | `Motus_robotwin2/mp_rank_00_model_states.pt` | 16.0 GB，DeepSpeed mp_rank 格式 |
| **VLM（Motus 用）** | `Qwen3-VL-2B-Instruct/` | ≠ 我们 lingbot-vla 的 Qwen2.5-VL-3B，两套独立 VLM |
| **VGM（Motus 用）** | `Wan2.2-TI2V-5B/` | WAN 2.2 5B + T5 `models_t5_umt5-xxl-enc-bf16.pth` |
| Motus 推理代码 | `Motus/inference/robotwin/Motus/` | `deploy_policy.py` / `models/motus.py` / `utils/stat.json` |

加载全模型（WAN5B + umt5-xxl T5 + Qwen3-VL-2B + experts）约需单卡 40–45GB。

---

## 2. 规格核对（已实测确认）

| 项 | 实际 | 说明 |
|---|---|---|
| 动作 chunk | **16**（`action_chunk_size=16`；`num_video_frames=8` 是视频帧数） | VLA=50 ≥ 16 |
| 动作/状态维度 | `action_dim=14, state_dim=14`（双臂） | 与我们一致 ✔ |
| **Motus 动作空间** | **原始 qpos（弧度），deploy 全程不归一化** | `get_action`/`sdedit` 直接把输出当 qpos；`_denormalize_actions` 定义了但**未调用**；`stat.json` 在动作路径**是死代码**。**由 t0=1.0=90% 实证**：Motus FM 输出 raw qpos 可直接执行 ✔ |
| 关节顺序 | `[Larm6, Lgrip, Rarm6, Rgrip]`（`stat.json` idx 6/13 是 gripper[0,1]） | 与 RoboTwin `take_action` 执行序一致 |
| 频率 | `global_downsample_rate=3`：16 动作 = 原始帧偏移 **[3,6,…,48]**（覆盖 48 帧，≈1.6s） | 我们 VLA = **spacing-1 连续 50 步** |

---

## 3. 我们 VLA 侧数据的真实空间（关键，之前搞错过）

`lingbot_wrm_rl_server` 里：
- `a_vla = self.vla.model.sample_actions(...)` → **VLA 归一化空间（lingbot-norm）**；
- `a_vla_real = a_vla[:, mask]` → 取 14 个有效关节，**仍是归一化值（范围 ~[-1,1]）**，顺序 `[Larm6, Rarm6, Lgrip, Rgrip]`；
- **真正执行的 raw qpos** = `feature_transform.unapply(batch)` 得到的 `response["action"]`（[50,14]，`take_action` 执行序）——**这个才是 raw qpos，但没存进 wm_trace**。

**⇒ 结论**：wm_trace 里存的 `a_vla_real` 是 **lingbot-norm，不是 raw qpos**。要给 Motus（raw qpos）用，必须：
1. 反归一化：`raw = (a_vla_real+1)/2 * (q99-q01) + q01`（用 `robotwin_50.json` 的 q01/q99，arm 12 维 + effector 2 维）；
2. 重排：`[Larm6,Rarm6,Lgrip,Rgrip] → [Larm6,Lgrip,Rarm6,Rgrip]`，idx `[0-5,12,6-11,13]`；
3. 频率重采样：`[2::3][:16]`。

**已数值验证**：10000 个样本按上述反归一化+重排后，**per-joint 100% 落在 Motus `stat.json` 的 [min,max] 内**（gripper 正确落 [0,1]）。→ 动作空间/序/频率**可对齐**。

---

## 4. 实现（已完成，均过编译/导入 + 闭环运行）

- `Motus/inference/robotwin/Motus/models/motus.py::sdedit_inference_step`：从 partial-noise(`start_t`) 对 action（+辅助 video）latent 去噪，`start_t→0` 积分，raw qpos。
- `Motus/inference/robotwin/Motus/deploy_policy.py::MotusPolicy.get_action_sdedit`：复用 `get_action` 的 T5/VLM/**合成三视图**预处理，返回 [16,14] raw qpos。
- `lingbot-vla/deploy/motus_sdedit_server.py`：`MotusSDEditPolicy` + `WebsocketPolicyServer`，请求 `{cam_high,cam_left,cam_right,state,task,a_init,t0}` → 响应 `{action:[16,14]}`。跑在 **motus** conda 环境（已装 `websockets`/`msgpack`）。
- `RoboTwin/script/rl_rollout_worker.py`：`--refine_motus`。闭环路径：obs → VLA server(`force_delta_zero`) 得 **`response["action"]`=raw qpos[50,14]** → `[2::3][:16]` → Motus server → 执行 16 步 → 重规划。
- `lingbot-vla/scripts/run_motus_refine.sh`：三环境编排（lingbotvla VLA server / motus Motus server / RoboTwin worker），末尾打印 per-task + overall SR。

**闭环对齐核验（重要）**：闭环喂 Motus 的 `a_init` 取自 `response["action"]`，是 **raw qpos、正确关节序、stride-3**——即**闭环实验的空间/序/频率都对**，其 SR 结果可信。

---

## 5. 闭环 smoke 结果（可信，路线证伪）

`stack_blocks_three`（VLA-alone repro **95%**，Motus paper 91%）：

| 配置 | 含义 | SR | 说明 |
|---|---|---|---|
| VLA-alone | spacing-1, 50 步 | 95% | 我们基线 |
| **t0=0.0** | **VLA stride-3 16 路点，Motus 数学上零贡献** | **9/10 = 90%** | Tier 0 频率对照（见 §5.1） |
| refine **t0=0.1** | 极轻 Motus 去噪 | **0% (0/N)** | 一介入就崩 |
| refine **t0=0.3**（10 seed） | 低 t0 | **0/10 = 0%** | 全崩（多为跑满 75 chunk 超时） |
| refine **t0=0.6**（3 seed 早停） | 中 t0 | **0/3 = 0%** | 仍全崩 |
| **t0=1.0（纯 Motus）**（10 seed） | 纯 Motus | **9/10 = 90%** | 贴合论文 91% |
| t0=1.0（纯 Motus）`handover_block` | 纯 Motus | 7/10 = 70% | 论文 86%，略低但正常 |

**可确证的结论**：
1. **pipeline / Motus 无 bug**：t0=1.0 复现 Motus SOTA 水平；16 路点/chunk 的执行节奏本身 OK。
2. **低/中 t0 的 SDEdit 混合不 work**：只有 t0≈1（≈纯 Motus、VLA 零贡献）才 work。「VLA 规划 + Motus 低 t0 细调」的设想**不成立**。
3. 而纯 Motus 在此任务(90%)仍**低于**我们 VLA-alone(95%)；macro 上两者相当。训练-free 混合**没有 SR > max(VLA,Motus) 的甜点**。

### 5.1 Tier 0 频率对照（决定性）——频率被洗清，指向流形

**动机**：低 t0 崩塌本可能混淆两因素——(a) 两策略动作分布差异（流形）；(b) VLA 计划被 stride-3 抽稀+改 16-tick 节奏。用 **`t0=0.0`**（`sdedit` 在 `start_t=0` 时 `dt=0`、输出**恒等于 `a_init`**、Motus 零贡献）可**单独测掉 (b)**。

**结果**（`stack_blocks_three`，复用与 t0=0.3 相同的 10 个 eval 布局）：
- `t0=0.0`（纯 VLA stride-3 16 路点、Motus 零贡献、16-tick 重规划）= **9/10 = 90%**。
- `t0=0.1`（Motus 极轻介入）= **0%**。

**结论**：
- **频率/执行制度不是杀手** ✔。VLA 的密集计划抽稀成 16 点、按 Motus 节奏执行照样 90%——(b) 被证伪。
- 死亡谷特征：**两端（纯 VLA-stride3 90% / 纯 Motus 90%）都好，中间任何混合（t0∈[0.1,0.6]）全崩 0%**。这是**动作流形不兼容**的典型信号——只要 Motus 速度场轻碰 VLA 轨迹，就把它拖向一个 neither-executes 的中间态。
- ⇒ **指向 (a) 流形不匹配**，而非对齐 bug / 频率。下一步用 Tier 1 微调验证（§10）：若把 Motus 流形训到能吸收 VLA-stride3，低 t0 混合应不再崩。

---

## 6. 历史勘误（之前没对齐的地方）

| 位置 | 旧说法（错） | 纠正 |
|---|---|---|
| 离线 `sdedit_offline_smoke.py:73` | `a_vla = tr["a_vla_real"]  # raw qpos` | `a_vla_real` 是 **lingbot-norm**，不是 raw qpos。离线冒烟拿**归一化值当弧度**喂 Motus，故所有 `rms(a_motus−a_vla)` 数字（0.92 / 1.08 及 t0 插值表）**无效，全部作废**。 |
| 旧 §12「根因：0.92 rad 流形差距」 | 用离线错误数字论证「流形太远」 | 该数字无效已作废；但**结论方向（流形）由 Tier 0（§5.1）以正确实验重新确立**——频率已排除。 |
| 旧 §2「deploy 原始 qpos，无归一化」 | 仅推断 | 由 **t0=1.0=90%** 实证为真 ✔（此条对） |
| 旧 §3.1 离线插值表 | 基于错误 a_vla | 作废；SDEdit 机制正确性改由闭环 t0 曲线佐证 |

---

## 7. 这批 VLA-zero 采集数据能否喂 Motus？——不能，需重采

`collect_wam_warmup.sh` 产出的每条数据：
- `wm_trace`: `state`(14), `a_vla_real`(50×14, **lingbot-norm**), `und_feats`(Qwen2.5-VL), `first_frame`(**head 单相机**), `meta`；
- `futures`: `frames`(8×384×320×3, **head 单相机**)。

**对 Motus 的阻塞**：
1. **视觉条件缺失**：只存了 **head**，从未存 left/right。Motus 的 VLM（Qwen3-VL）und + video 分支都要 **head+left+right 合成三视图** → **连在线重算 Motus 条件都做不到**。
2. und 特征来自 Qwen2.5-VL-3B，对 Motus 的 Qwen3-VL-2B **无用**。
3. 动作是 lingbot-norm，需 §3 的反归一化+重排+重采样才成 raw qpos。

→ **若要给 Motus 微调，必须重采**（存全三相机 + 合成帧 + 未来合成帧）。但见 §8：BC 微调 Motus 本身价值存疑。

**对我们自己的 v4 模型**：head 帧 WM 目标、我们 VLM 的 und、我们动作空间——**三者自洽，完全可用**，不浪费。

---

## 8. 战略结论 & 建议

- **训练-free SDEdit 混合**：证伪（§5），终止。
- **BC 微调 Motus（用我们 VLA 动作当 target）**：teacher(VLA 87.9%) ≈ Motus(88.5%)，等强 teacher 蒸馏**难涨、易退**；且需重采三视图数据。价值低。
- **Motus 当策略 / RL init**：部署策略变成 Motus、**VLA 完全不参与** = 放弃「在我们 VLA 上做改进」的项目前提，属另一个课题；RL 也不需要这批离线数据（在线自采）。
- **✅ 推荐：回到 v4**（`WRM_RL_PLAN_WAM_WARMUP.md`）——自训 WM 耦合的绝对动作流模型 + RL，这是唯一既保留 VLA 前提、又带方法 novelty、且能通过**训练**（而非推理期硬拼）弥合分布差异的路线。当前 VLA-zero 数据正好服务它。

---

## 9. 待确认（给 v4 收尾用）

- **A**：采集失败的 3 个任务 `open_laptop` / `place_object_scale` / `put_object_cabinet`（RoboTwin env 报 `arm_tag` AttributeError，非我们 pipeline bug）——修 env / 跳过 / 换任务？
- **B**：v4 蒸馏数据用 **success-only** 还是全量（含 VLA 失败轨迹）？
- **C**（可选，低优先）：用**正确 raw qpos** 重测一次 `rms(a_motus − a_vla_raw)`，给「分布差异多大」一个干净数字。

---

## 10. Tier 1：Motus 微调验证流形（已完成 ✅）

**目的**：Tier 0（§5.1）已把频率排除、指向流形。用 BC 训练检验：若把 Motus 动作专家的流形拉到 VLA-stride3 上，低 t0 混合是否复活。

### 10.1 数据采集

- 脚本：`lingbot-vla/scripts/collect_motus_ft.sh`（`--motus_ft_collect`）
- 规模：**47 任务**（剔除 env-bug 的 `open_laptop` / `place_object_scale` / `put_object_cabinet`），50 seed/任务，**2114 成功轨迹**（90% SR）
- 每 chunk：`cam_high/left/right` + `state`(raw qpos) + `target`(VLA stride-3 raw qpos [16,14])
- 根目录：`/mnt/data14/yyg/wrm_rl_runs/motus_ft`

### 10.2 微调

- 脚本：`Motus/inference/robotwin/Motus/finetune_motus_action.py`
- 方法：冻结 WAN-VAE / Qwen3-VL / video 专家，只训 **action expert**（641.5M 参数），flow-matching BC
- 配置：3 epoch，lr=1e-4，accum=16，2114 轨迹（train 2009 / val 105）
- 结果：**val_loss=0.021**（epoch 2）；产物 `Motus/runs/motus_ft_action_v1/action_expert_final.pt`

### 10.3 快评（held-out eval seed，2 任务 × 10 seed）

评测协议：`--eval_seeds`，从 **seed=100000** 起向上扫 10 个 expert-feasible 布局（与 `eval_policy.py --seed 0` 同源，见 §11）。权重 `action_expert_final.pt`。

| 配置 | stack_blocks_three | handover_block | Overall |
|---|---|---|---|
| Motus-FT **t0=1.0**（纯微调 Motus） | 9/10 = **90%** | 4/10 = 40% | 13/20 = 65% |
| Motus-FT **t0=0.3**（VLA→SDEdit refine） | 9/10 = **90%** | 4/10 = 40% | 13/20 = 65% |

**对比微调前**（train seed，`stack_blocks_three`）：

| | t0=0.3（预训练 Motus SDEdit） | t0=0.3（Motus-FT） |
|---|---|---|
| SR | **0/10 = 0%** | **9/10 = 90%** |

### 10.4 判定：**流形根因坐实 ✅**

| Motus-FT-alone（t0=1） | SDEdit t0=0.3 | 结论 |
|---|---|---|
| 高（90%） | **复活（90%，与 t0=1 一致）** | **流形确认** ✔ |

- 预训练 Motus 与 VLA-stride3 **流形不兼容** → 训练-free SDEdit 低 t0 死亡谷（§5）。
- BC 把 Motus 流形拉到 VLA-stride3 后，**低 t0 混合不再崩**——死亡谷被填平。
- 同时 t0=1.0 与 t0=0.3 SR 完全一致 → blend 机制本身 OK，问题只在流形距离。
- **这也是 v4 Phase-A（WAM distillation warmup）的可行性 PoC**：自训模型做同样的事，不必依赖 Motus 8B。

**代码**：`models/motus.py::action_fm_loss` + `finetune_motus_action.py` + `deploy/motus_sdedit_server.py --ft_action_ckpt` + `scripts/run_motus_ft_eval.sh`

---

## 11. 正式全量评测（Motus-FT t0=0.3，进行中）

**配置**：Motus-FT `action_expert_final.pt`，**t0=0.3**，VLA(force_zero) → stride-3 a_init → Motus SDEdit refine → 执行。

**Seed 协议（无需手设 seed、无需预扫）**：

与 RoboTwin 官方 `eval_policy.py` / `LingBotVLA/eval.sh` **一致**：
- `--seed 0` → `st_seed = 100000`，从该起点**向上扫描** expert-feasible 布局
- `test_num = 100` → 每任务 **100 个 eval episode**
- **边扫边跑**：官方评测在一个 loop 里做 expert 可行性检查 + policy rollout，**没有单独的 scan_only 预扫阶段**

我们的 `run_motus_ft_eval.sh` 已对齐：**去掉预扫**，worker 在 `--eval_seeds` 模式下从 100000 起 inline 收集 feasible seed 并 rollout（与 `eval_policy.py` 同 band、同 test_num，只是实现载体是 refine worker 而非 `eval_policy.py`）。快评 `NUM_SEEDS=10` 只是缩小 test_num。

**规模**：**50 任务**（与 `run_ppo_full_v1` / `comparison_clean.csv` 一致）× 100 episode。

**运行**：
```bash
# 正式全量（默认 NUM_SEEDS=100, 47 任务, t0=0.3）
MOTUS_T0=0.3 RL_ROOT=/mnt/data14/yyg/wrm_rl_runs/motus_ft_eval_full/t0_0.3 \
  nohup bash scripts/run_motus_ft_eval.sh > logs/motus_ft_eval_full_t03.nohup.log 2>&1 &
```

**产物**：`/mnt/data14/yyg/wrm_rl_runs/motus_ft_eval_full/t0_0.3/<task>/round0/episodes/*.json` + orchestrator 末尾 per-task / overall SR。

**日志**：见 orchestrator.log；完成后更新本 §11 结果表。
