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

import torch


# Import flash attention cross entropy for efficient logprobs computation
try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    FLASH_ATTN_CROSS_ENTROPY_AVAILABLE = True
except ImportError:
    FLASH_ATTN_CROSS_ENTROPY_AVAILABLE = False
    print("flash_attn cross_entropy_loss not available, falling back to torch.log_softmax for logprobs")


def logprobs_from_logits_flash_attn(
    logits: torch.Tensor, labels: torch.Tensor, inplace_backward: bool = False
) -> torch.Tensor:
    """
    Compute log probabilities from logits using flash attention cross entropy.

    Args:
        logits: Logits tensor of shape [batch_size, vocab_size]
        labels: Label tensor of shape [batch_size] containing token IDs
        inplace_backward: Whether to use inplace backward (default: False for inference)

    Returns:
        Log probabilities tensor of shape [batch_size] for the selected tokens
    """
    assert FLASH_ATTN_CROSS_ENTROPY_AVAILABLE
    if not FLASH_ATTN_CROSS_ENTROPY_AVAILABLE:
        # Fallback to standard torch implementation
        log_probs = torch.log_softmax(logits, dim=-1)
        # Gather log probs for selected tokens
        return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    # Use flash attention cross entropy for efficient computation
    output = cross_entropy_loss(logits, labels, inplace_backward=inplace_backward)
    assert isinstance(output, tuple), (
        "please make sure flash-attn>=2.4.3 where cross_entropy_loss returns Tuple[losses, z_losses]."
    )
    # Return negative of losses to get log probabilities
    return -output[0]
