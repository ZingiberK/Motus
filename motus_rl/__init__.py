"""Phase 2 Motus RL: Flow-SDE PPO on the action expert + supervised World Model.

See ``Motus/MOTUS_PLAN.md`` (Phase 2). Isolated training:
video (WAN) branch = supervised flow-matching on simulator-observed futures;
action expert = PPO on Flow-SDE denoise log-probs (never mixes gradients).
"""
