set -ex

if [ -z "$model_dir" ]; then
    echo "Error: model_dir is not set" >&2
    exit 1
fi
ls $model_dir

ATTN_IMPL=${ATTN_IMPL:-flash_attention_3}
INFER_FA_IMPL=${INFER_FA_IMPL:-fa-invariant}
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

# Disable Liger kernels for VeOmni
export VEOMNI_USE_LIGER_KERNEL=0

# Clean stale outputs from any previous invocation so verify never reads
# leftover tensors from a different model/attn configuration.
rm -rf inference_outputs inference_outputs_rand inference_outputs_rand_pp

# Case 1: 10 identical long prompts without chunked prefill
# self-consistent invariance with continuous batching and randomly assigned paged block order and use fp32 logits
CUDA_VISIBLE_DEVICES=0 python tests/scripts/hf_inference.py --model_path $model_dir --simulate_requests 10 --max_num_batched_tokens 2048 --request_interval 0.01 --max_length 256 --max_new_tokens 256 --enable_batch_invariant --output_dir inference_outputs --use_fp32_logits --attn_impl $INFER_FA_IMPL
# vs native hf, correspondingly we need to enable fused lce to compute logprobs with fp32 logits
CUDA_VISIBLE_DEVICES=0 python tests/scripts/verify_logits_vs_native_hf.py  --model_path $model_dir --attn_impl $ATTN_IMPL --enable_batch_invariant --data_dir inference_outputs --use_fused_lce

# # Case 2: 10 random prompts with chunked prefill
CUDA_VISIBLE_DEVICES=0 python tests/scripts/hf_inference.py --model_path $model_dir --simulate_requests 10 --max_num_batched_tokens 2048 --request_interval 0.01 --max_length 1024 --max_new_tokens 1024 --enable_batch_invariant --output_dir inference_outputs_rand --simulate_requests_random_contents --enable_chunked_prefill --use_fp32_logits --attn_impl $INFER_FA_IMPL
# Check padded input: flash_attn_with_kvcache vs flash_attn_func
CUDA_VISIBLE_DEVICES=0 python tests/scripts/verify_logits_vs_native_hf.py  --model_path $model_dir --attn_impl $ATTN_IMPL --enable_batch_invariant --data_dir inference_outputs_rand --use_fused_lce
# Check remove pad inputs: flash_attn_with_kvcache vs flash_attn_varlen_func
CUDA_VISIBLE_DEVICES=0 python tests/scripts/verify_logits_vs_native_hf.py  --model_path $model_dir --attn_impl $ATTN_IMPL --enable_batch_invariant --data_dir inference_outputs_rand --use_remove_padding --use_fused_lce


# Case 3: HF Inference Pipeline Parallelism (uses its own output dir to avoid
# clobbering Case 2 artifacts).
if [ "$NUM_GPUS" -ge 2 ]; then
    python tests/scripts/hf_inference.py --model_path $model_dir --simulate_requests 10 --max_num_batched_tokens 2048 --request_interval 0.01 --max_length 1024 --max_new_tokens 1024 --enable_batch_invariant --output_dir inference_outputs_rand_pp --simulate_requests_random_contents --enable_chunked_prefill --use_fp32_logits --pipeline_parallel_size 2 --attn_impl $INFER_FA_IMPL
    CUDA_VISIBLE_DEVICES=0 python tests/scripts/verify_logits_vs_native_hf.py  --model_path $model_dir --attn_impl $ATTN_IMPL --enable_batch_invariant --data_dir inference_outputs_rand_pp --use_remove_padding --use_fused_lce
else
    echo "Skipping pipeline parallelism test (requires 2+ GPUs, found $NUM_GPUS)"
fi
