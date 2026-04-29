# Copyright 2025 Bytedance Ltd. and/or its affiliates

"""Register VeXact rollout with VeRL framework.

Usage:
    Set environment variable before importing verl:
        export VERL_USE_EXTERNAL_MODULES=vexact.integrations.verl.register

    Or import this module directly after verl:
        import verl
        import vexact.integrations.verl.register
"""

from verl.workers.rollout.base import _ROLLOUT_REGISTRY
from verl.workers.rollout.replica import RolloutReplicaRegistry


def _load_vexact_replica():
    """Lazy loader for VeXactReplica to avoid circular imports."""
    from vexact.integrations.verl.async_server import VeXactReplica

    return VeXactReplica


# Register VeXact rollout replica (for server mode)
RolloutReplicaRegistry.register("vexact", _load_vexact_replica)

# Register VeXact rollout base (for hybrid mode with device mesh)
_ROLLOUT_REGISTRY[("vexact", "async")] = "vexact.integrations.verl.rollout.ServerAdapter"

print(f"[vexact] Registered VeXact with VeRL at {__file__}")
