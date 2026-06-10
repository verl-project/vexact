#!/usr/bin/env python3
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

"""
Test script for load_state_dict functionality.

This script tests loading a state dict into DriverWorker and verifying
that the weights are actually updated.
"""

import os
import uuid

import torch
from transformers import GenerationConfig

from tests.conftest import get_tests_attn_impl
from vexact.config import ModelConfig, ParallelConfig, SchedulerConfig, VeXactConfig
from vexact.core.request import InferenceRequest
from vexact.utils.tokenizer import load_tokenizer
from vexact.worker.driver_worker import DriverWorker


def test_load_state_dict():
    """Test loading state dict by comparing generation outputs."""
    model_path = os.environ["VEXACT_TESTS_MODEL_PATH"]

    config = VeXactConfig(
        model=ModelConfig(
            model_path=model_path,
            attn_impl=get_tests_attn_impl(),
            enable_batch_invariant=True,
        ),
        parallel=ParallelConfig(pipeline_parallel_size=1),
        scheduler=SchedulerConfig(max_num_batched_tokens=8, enable_chunked_prefill=True),
    )

    worker = DriverWorker(config)
    worker.start()
    tokenizer = load_tokenizer(model_path)

    def generate(input_ids, gen_config):
        request = InferenceRequest(
            request_id=f"req_{uuid.uuid4().hex[:8]}",
            generation_config=gen_config,
            input_ids_list=input_ids,
        )
        worker.submit_request(request)
        results = worker.poll_results(timeout=60.0)
        return results[0] if results else None

    try:
        print("=" * 60)
        print("Testing load_state_dict()")
        print("=" * 60)

        # Prepare test input
        print("\n1. Preparing test input...")
        test_prompt = "What is AI?"
        input_ids = tokenizer.encode(test_prompt, add_special_tokens=True)
        print(f"   Test prompt: {test_prompt}")
        print(f"   Input length: {len(input_ids)}")

        # Setup generation config (greedy decoding for determinism)
        gen_config = GenerationConfig(
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Get original state dict
        print("\n2. Saving original state dict...")
        original_state_dict = {k: v.detach().clone() for k, v in worker.model.state_dict().items()}
        print(f"   State dict has {len(original_state_dict)} parameters")

        # Generate with original weights
        print("\n3. Generating with original weights...")
        result = generate(input_ids, gen_config)
        original_tokens = result.generated_tokens
        original_text = tokenizer.decode(original_tokens, skip_special_tokens=True)
        print(f"   Generated tokens: {original_tokens}")
        print(f"   Generated text: {original_text}")

        # Create a modified state dict (add noise to weights)
        print("\n4. Creating modified state dict with noise...")
        modified_state_dict = {}
        for name, param in original_state_dict.items():
            noise = torch.randn_like(param) * 0.01
            modified_state_dict[name] = (param + noise).detach().clone()
        print("   Added noise with std=0.01 to all parameters")

        # Load modified state dict
        print("\n5. Loading modified state dict...")
        res = worker.load_state_dict(modified_state_dict)
        print(f"   Load result: {res}")

        # Generate with modified weights
        print("\n6. Generating with modified weights...")
        result = generate(input_ids, gen_config)
        modified_tokens = result.generated_tokens
        modified_text = tokenizer.decode(modified_tokens, skip_special_tokens=True)
        print(f"   Generated tokens: {modified_tokens}")
        print(f"   Generated text: {modified_text}")

        # Verify outputs differ from original
        print("\n7. Verifying outputs differ from original...")
        tokens_differ = original_tokens != modified_tokens
        print(f"   Tokens differ: {tokens_differ}")

        assert tokens_differ, "Outputs didn't change after weight update!"

        # Restore original weights
        print("\n8. Restoring original weights...")
        res = worker.load_state_dict(original_state_dict)
        print(f"   Load result: {res}")

        # Generate with restored weights
        print("\n9. Generating with restored weights...")
        result = generate(input_ids, gen_config)
        restored_tokens = result.generated_tokens
        restored_text = tokenizer.decode(restored_tokens, skip_special_tokens=True)
        print(f"   Generated tokens: {restored_tokens}")
        print(f"   Generated text: {restored_text}")

        # Verify outputs exactly match original
        print("\n10. Verifying outputs exactly match original...")
        tokens_match = original_tokens == restored_tokens
        print(f"   Tokens exactly match: {tokens_match}")

        if not tokens_match:
            print(f"   Original tokens:  {original_tokens}")
            print(f"   Restored tokens:  {restored_tokens}")

        assert tokens_match, "Restored weights don't produce identical outputs!"
        print("   ✓ Restored weights produce identical outputs!")

        print("\n" + "=" * 60)
        print("✓ Test completed successfully!")
        print("=" * 60)

    finally:
        worker.stop()
