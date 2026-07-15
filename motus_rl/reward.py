"""Reward / advantage helpers for Motus Flow-SDE PPO (Phase 2).

Terminal success sparse reward, discounted return-to-go per action chunk,
optional paired baseline (success_policy - success_ref) for Motus-vs-VLA contrast.
"""

from __future__ import annotations

from typing import List


def episode_returns(success: bool, n_chunks: int, gamma: float = 0.99) -> List[float]:
    """``return_to_go[t] = gamma**(T-1-t) * float(success)``."""
    if n_chunks <= 0:
        return []
    s = float(bool(success))
    T = int(n_chunks)
    return [(gamma ** (T - 1 - t)) * s for t in range(T)]


def paired_baseline_returns(
    success_policy: bool,
    success_ref: bool,
    n_chunks: int,
    gamma: float = 0.99,
) -> List[float]:
    """Paired baseline: ``A_t ∝ (success_policy − success_ref)``.

    Use when each (task, seed) has a Motus-SDE episode and a VLA-only (or Motus
    t0=0) reference episode. Positive when Motus beats the reference.
    """
    if n_chunks <= 0:
        return []
    outcome = float(bool(success_policy)) - float(bool(success_ref))
    T = int(n_chunks)
    return [(gamma ** (T - 1 - t)) * outcome for t in range(T)]


paired_vla_returns = paired_baseline_returns
