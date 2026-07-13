"""Train / eval seed isolation for WRM Flow-GRPO.

Fairness hard-constraint (WRM_RL_PLAN.md §0): RL may only interact with TRAIN
seeds; the 100-episode evaluation must run on a DISJOINT band of EVAL seeds.

RoboTwin's evaluation (`RoboTwin/script/eval_policy.py`) starts from
``st_seed = 100000 * (1 + seed)`` and scans *upward*, accepting only
expert-solvable seeds until it has collected ``test_num`` (=100) episodes. With
the conventional ``--seed 0`` this means eval consumes seeds starting at
100000. We therefore reserve a wide EVAL band ``[100000, 100000 + EVAL_BAND)``
and draw TRAIN seeds from ``[0, 100000)`` — guaranteed disjoint because a single
task only ever needs a few dozen feasible train seeds.
"""

from __future__ import annotations

EVAL_BASE = 100_000          # RoboTwin eval st_seed for --seed 0
EVAL_BAND = 100_000          # reserve [100000, 200000) for evaluation, never trained on
TRAIN_BASE = 0               # train seeds scanned upward from here


def is_eval_seed(seed: int) -> bool:
    """True if ``seed`` falls inside the reserved evaluation band."""
    return EVAL_BASE <= seed < EVAL_BASE + EVAL_BAND


def is_train_seed(seed: int) -> bool:
    """True if ``seed`` is a legal training seed (outside the eval band)."""
    return not is_eval_seed(seed)


def train_seed_candidates(num: int, base: int = TRAIN_BASE):
    """Yield ``num`` candidate train seeds starting at ``base``.

    These are *candidates*; the rollout worker still runs RoboTwin's expert
    check and keeps only feasible (expert-solvable) initial states, mirroring
    ``collect_lingbot_vla_recovery_rollouts.py``.
    """
    seed = base
    yielded = 0
    while yielded < num:
        if is_train_seed(seed):
            yield seed
            yielded += 1
        seed += 1


def assert_disjoint(train_seeds, eval_seeds=None) -> None:
    """Raise if any train seed leaks into the eval band / eval set."""
    leaked = [s for s in train_seeds if is_eval_seed(s)]
    if leaked:
        raise ValueError(
            f"{len(leaked)} train seed(s) fall in the reserved eval band "
            f"[{EVAL_BASE}, {EVAL_BASE + EVAL_BAND}): e.g. {leaked[:5]}"
        )
    if eval_seeds is not None:
        overlap = sorted(set(train_seeds) & set(eval_seeds))
        if overlap:
            raise ValueError(f"train/eval seed overlap: {overlap[:5]} ...")
