"""Reward and GRPO advantage for WRM Flow-GRPO (v1, intentionally simple).

v1 reward (WRM_RL_PLAN.md §9.2):

    R = success - lambda_delta * mean_chunk( rms(Delta) )

where ``success`` is RoboTwin's terminal success (1.0 / 0.0), ``Delta`` is the
WRM residual for one action chunk in *LingBot-normalized* action space, and
``rms(Delta) = sqrt(mean(Delta**2))`` is a scale-interpretable per-element
magnitude. The single magnitude term gives the desired asymmetric optimum
automatically: when the VLA is already correct, Delta=0 is optimal (success is
reachable for free, any residual only costs penalty); when the VLA fails, the
optimum is the *smallest* residual that flips failure -> success. No dense /
direction / KL-to-zero terms in v1 (keep it simple, tune ``lambda_delta`` only).

GRPO advantage: group-relative normalization over the G rollouts sharing one
train seed, broadcast to every chunk (and every denoise step) of a rollout.
"""

from __future__ import annotations

import math
from typing import List, Sequence

DEFAULT_LAMBDA_DELTA = 0.05   # small on purpose; primary tuning knob
ADV_EPS = 1e-4                # std floor for group normalization


def chunk_rms(delta_flat_sq_sum: float, n_elem: int) -> float:
    """RMS magnitude of one chunk's residual given sum-of-squares and #elements."""
    if n_elem <= 0:
        return 0.0
    return math.sqrt(max(delta_flat_sq_sum, 0.0) / n_elem)


def rollout_reward(
    success: bool,
    chunk_rms_list: Sequence[float],
    lambda_delta: float = DEFAULT_LAMBDA_DELTA,
) -> float:
    """R = success - lambda_delta * mean_chunk(rms(Delta))."""
    mean_mag = (sum(chunk_rms_list) / len(chunk_rms_list)) if chunk_rms_list else 0.0
    return float(success) - float(lambda_delta) * float(mean_mag)


def group_advantages(rewards: Sequence[float], eps: float = ADV_EPS) -> List[float]:
    """Group-relative advantages: A_i = (R_i - mean) / (std + eps).

    A degenerate group (all rewards equal, e.g. all-success or all-fail) yields
    ~zero advantages, which is exactly the desired "nothing to learn here"
    behavior (see WRM_RL_PLAN.md §8.3).
    """
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(var)
    denom = std + eps
    return [(r - mean) / denom for r in rewards]


# ---------------------------------------------------------------- PPO (v2)
# Pure sparse success reward (aligned with RLinf env/success_once): the only
# reward is delivered at the terminal chunk = float(success). We treat each
# action chunk as one MDP step and discount the terminal reward back with gamma.
# The critic V(s_t) provides the baseline, so the advantage is
#     A_t = return_to_go_t - V(s_t),   return_to_go_t = gamma^(T-1-t) * success
# i.e. GAE with lambda=1 (Monte-Carlo return, value baseline). This needs V only
# at the chunk itself (no neighbour bootstrapping) -> we never have to load the
# non-subsampled chunks' traces, keeping the learner's disk I/O bounded. The
# value target (return) is return_to_go_t; PPO value-clips V_new around V_old.

def episode_returns(success: bool, n_chunks: int, gamma: float = 0.99) -> List[float]:
    """Discounted return-to-go per chunk for a terminal-only success reward.

    ``return_to_go[t] = gamma**(T-1-t) * float(success)`` for t in [0, T).
    """
    if n_chunks <= 0:
        return []
    s = float(bool(success))
    T = int(n_chunks)
    return [(gamma ** (T - 1 - t)) * s for t in range(T)]


def paired_vla_returns(
    success_wrm: bool,
    success_vla: bool,
    n_chunks: int,
    gamma: float = 0.99,
) -> List[float]:
    """Paired VLA baseline: A_t = gamma^(T-1-t) * (success_wrm - success_vla).

    Used by PPO v3 so WRM episodes get positive advantage when WRM beats its
    paired delta=0 arm, and negative when VLA-only was better.
    """
    if n_chunks <= 0:
        return []
    outcome = float(bool(success_wrm)) - float(bool(success_vla))
    T = int(n_chunks)
    return [(gamma ** (T - 1 - t)) * outcome for t in range(T)]
