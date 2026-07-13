"""Motus Flow-SDE PPO + supervised World Model (adapted from WRM ``wrm_rl``).

See ``Motus/MOTUS_RL_PLAN_PPO.md``. Video branch = supervised FM on observed
futures; action branch = PPO on Flow-SDE log-probs. Original WRM learner kept
under ``legacy/wrm_rl/`` for reference.
"""
