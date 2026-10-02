# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Initialize VeOmni state for the standalone rollout process group."""

import torch.distributed as dist


def init_veomni_parallel_state():
    """Bind a non-EP state using the current VeOmni config-based API."""
    if not dist.is_initialized():
        return

    from veomni.arguments import AcceleratorConfig
    from veomni.distributed.parallel_state import init_parallel_state_from_config

    accelerator = AcceleratorConfig()
    # AcceleratorConfig derives these attributes from WORLD_SIZE, which may
    # describe the parent training job rather than this rollout process group.
    world_size = dist.get_world_size()
    accelerator.world_size = world_size
    accelerator.dp_size = world_size
    accelerator.dp_shard_size = world_size
    return init_parallel_state_from_config(accelerator, name=None)
