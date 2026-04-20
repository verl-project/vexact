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

import multiprocessing as mp
import time

import pytest
import torch


pytest.importorskip("vexact.integrations.verl.bucketed_weight_transfer")

from vexact.integrations.verl.bucketed_weight_transfer import BucketedWeightReceiver, BucketedWeightSender  # noqa: E402


def create_test_weights():
    """Create test weights with known values."""
    torch.manual_seed(42)
    return {
        "layer1.weight": torch.randn(256, 256, dtype=torch.float32, device="cuda:0"),
        "layer2.weight": torch.randn(512, 512, dtype=torch.bfloat16, device="cuda:0"),
        "layer3.weight": torch.randn(1024, 1024, dtype=torch.float32, device="cuda:0"),
    }


def compute_checksums(state_dict):
    """Compute checksums for verification."""
    return {name: (tensor.sum().item(), tensor.shape, tensor.dtype) for name, tensor in state_dict.items()}


def sender_process(zmq_handle: str, checksums_queue: mp.Queue):
    """Send weights and report checksums."""
    weights = create_test_weights()
    checksums = compute_checksums(weights)

    print(f"[Sender] Sending {len(weights)} tensors...")
    for name, (sum_val, shape, dtype) in checksums.items():
        print(f"  {name}: shape={shape}, dtype={dtype}, sum={sum_val:.4f}")

    sender = BucketedWeightSender(zmq_handle, bucket_size_mb=512)

    def weight_generator():
        yield from weights.items()

    start = time.time()
    sender.send_weights(weight_generator())
    elapsed = time.time() - start

    print(f"[Sender] Done in {elapsed * 1000:.1f}ms")

    # Send checksums to receiver for verification
    checksums_queue.put(checksums)


def receiver_process(zmq_handle: str, checksums_queue: mp.Queue):
    """Receive weights and verify against checksums."""
    time.sleep(0.5)  # Wait for sender to bind

    print("[Receiver] Receiving weights...")
    receiver = BucketedWeightReceiver(zmq_handle)
    received_weights = {}

    def on_bucket(weights):
        for name, tensor in weights:
            received_weights[name] = tensor

    start = time.time()
    receiver.receive_weights(on_bucket)
    elapsed = time.time() - start

    print(f"[Receiver] Received {len(received_weights)} tensors in {elapsed * 1000:.1f}ms")

    # Verify against sender checksums
    expected_checksums = checksums_queue.get()

    print("\n[Verification]")
    all_match = True
    for name, (expected_sum, expected_shape, expected_dtype) in expected_checksums.items():
        if name not in received_weights:
            print(f"  ✗ {name}: MISSING")
            all_match = False
            continue

        tensor = received_weights[name]
        actual_sum = tensor.sum().item()
        matches = (
            tensor.shape == expected_shape and tensor.dtype == expected_dtype and abs(actual_sum - expected_sum) < 1e-3
        )

        status = "✓" if matches else "✗"
        print(f"  {status} {name}: sum={actual_sum:.4f} (expected {expected_sum:.4f})")

        if not matches:
            all_match = False

    if all_match:
        print("\n✓ All tensors verified successfully!")
    else:
        print("\n✗ Verification failed!")
        raise AssertionError("Tensor verification failed")


def test_basic_transfer_with_verification():
    """Test basic transfer with verification."""
    print("=== Test: Basic Transfer with Verification ===\n")

    zmq_handle = "ipc:///tmp/test_weights_verify.sock"
    checksums_queue = mp.Queue()

    sender_proc = mp.Process(target=sender_process, args=(zmq_handle, checksums_queue))
    receiver_proc = mp.Process(target=receiver_process, args=(zmq_handle, checksums_queue))

    receiver_proc.start()
    sender_proc.start()

    sender_proc.join()
    receiver_proc.join()

    if receiver_proc.exitcode != 0:
        raise AssertionError("Receiver process failed")

    print("\n" + "=" * 50 + "\n")


def test_chunking():
    """Test automatic chunking for large tensors."""
    print("=== Test: Chunking Large Tensors ===\n")

    def sender_chunking(zmq_handle: str, checksums_queue: mp.Queue):
        # Create 256MB tensor that requires chunking with 64MB bucket
        torch.manual_seed(123)
        large_tensor = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda:0")
        checksum = (large_tensor.sum().item(), large_tensor.shape, large_tensor.dtype)

        print(f"[Sender] Sending large tensor: {large_tensor.shape} ({large_tensor.nbytes / (1024**2):.1f} MB)")
        print(f"  Checksum: sum={checksum[0]:.4f}")

        sender = BucketedWeightSender(zmq_handle, bucket_size_mb=64)  # Small bucket forces chunking

        def weight_gen():
            yield "large_embedding", large_tensor

        sender.send_weights(weight_gen())
        checksums_queue.put({"large_embedding": checksum})

    def receiver_chunking(zmq_handle: str, checksums_queue: mp.Queue):
        time.sleep(0.5)

        print("[Receiver] Receiving chunked tensor...")
        receiver = BucketedWeightReceiver(zmq_handle)
        received_tensor = None

        def on_bucket(weights):
            nonlocal received_tensor
            for _, tensor in weights:
                received_tensor = tensor

        receiver.receive_weights(on_bucket)

        expected_checksums = checksums_queue.get()
        expected_sum, expected_shape, expected_dtype = expected_checksums["large_embedding"]
        actual_sum = received_tensor.sum().item()

        print(f"[Receiver] Received: {received_tensor.shape} ({received_tensor.nbytes / (1024**2):.1f} MB)")
        print("\n[Verification]")
        print(f"  Shape: {received_tensor.shape} == {expected_shape}: {received_tensor.shape == expected_shape}")
        print(f"  Dtype: {received_tensor.dtype} == {expected_dtype}: {received_tensor.dtype == expected_dtype}")
        print(f"  Sum: {actual_sum:.4f} vs {expected_sum:.4f}, diff={abs(actual_sum - expected_sum):.6f}")

        if abs(actual_sum - expected_sum) > 1e-3:
            raise AssertionError("Chunked tensor verification failed")

        print("\n✓ Chunked tensor verified successfully!")

    zmq_handle = "ipc:///tmp/test_chunking_verify.sock"
    checksums_queue = mp.Queue()

    sender_proc = mp.Process(target=sender_chunking, args=(zmq_handle, checksums_queue))
    receiver_proc = mp.Process(target=receiver_chunking, args=(zmq_handle, checksums_queue))

    receiver_proc.start()
    sender_proc.start()

    sender_proc.join()
    receiver_proc.join()

    if receiver_proc.exitcode != 0:
        raise AssertionError("Receiver process failed")

    print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    test_basic_transfer_with_verification()
    test_chunking()

    print("\n✓ All tests passed!")
