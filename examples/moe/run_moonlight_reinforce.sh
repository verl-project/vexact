#!/bin/bash

# Example invocation:
#   VEXACT_MAX_CACHE_BLOCKS=1536 \
#   MODEL_PATH=/path/to/Moonlight-16B-A3B-Instruct \
#   DATA_TRAIN_PATH=/path/to/dapo_root \
#   DATA_TEST_PATH=/path/to/aime_root \
#   bash examples/moe/run_moonlight_reinforce.sh

set -x
model_path=${MODEL_PATH}
data_train_path=${DATA_TRAIN_PATH}
data_test_path=${DATA_TEST_PATH}
# Register vexact rollout globally
export VERL_USE_EXTERNAL_MODULES=vexact.integrations.verl.register
export VERL_LOGGING_LEVEL=DEBUG
export NCCL_DEBUG=ERROR
# VeOmni Liger Patch
export VEOMNI_USE_LIGER_KERNEL=0
# B200 (SM100+) uses FA4 CUTE kernel for batch-invariant inference
export INFER_FA_IMPL=${INFER_FA_IMPL:-fa-invariant-cute}
# Register and enable actor and ref model FSDP ops
verl_model_external_lib=vexact.integrations.verl.fsdp_enable_invariant
attn_implementation=${VEOMNI_ATTN_IMPL:-veomni_flash_attention_4_with_sp}
moe_implementation=${VEOMNI_MOE_IMPLEMENTATION:-"fused_quack"}
# Enable fused lce for both training and inference
use_fused_kernels=True
fused_kernel_backend=torch
# Use liger RMSNorm/RoPE/Swiglu
use_liger=False
vexact_max_cache_blocks=${VEXACT_MAX_CACHE_BLOCKS:-4096}
echo "${@:1}"
loss_agg_mode="seq-mean-token-sum-norm"

RAY_DEDUP_LOGS=1 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    model_engine=veomni \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files=$data_train_path/data/dapo-math-17k.parquet \
    data.val_files=$data_test_path/test/aime_2024.parquet \
    data.train_batch_size=512 \
    data.val_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=20480 \
    data.filter_overlong_prompts=True \
    data.seed=42 \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=$attn_implementation \
    actor_rollout_ref.model.external_lib=$verl_model_external_lib \
    actor_rollout_ref.model.use_fused_kernels=$use_fused_kernels \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=$fused_kernel_backend \
    actor_rollout_ref.model.use_liger=$use_liger \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=22528 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.policy_loss.loss_mode=gpg \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.actor.optim.betas="[0.9,0.95]" \
    +actor_rollout_ref.actor.optim.override_optimizer_config.eps=1e-15 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.veomni.fsdp_size=8 \
    actor_rollout_ref.actor.veomni.expert_parallel_size=1 \
    actor_rollout_ref.actor.veomni.attn_implementation=$attn_implementation \
    actor_rollout_ref.actor.veomni.moe_implementation=$moe_implementation \
    actor_rollout_ref.actor.veomni.param_offload=True \
    actor_rollout_ref.actor.veomni.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vexact \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    ++actor_rollout_ref.rollout.engine_kwargs.vexact.max_cache_blocks=$vexact_max_cache_blocks \
    ++actor_rollout_ref.rollout.engine_kwargs.vexact.attn_impl=$INFER_FA_IMPL \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    custom_reward_function.path=examples/math_reward_model/math_grader_moe.py \
    custom_reward_function.name=compute_math_score \
    trainer.project_name=vexact-baseline-math-moe-reinforce \
    trainer.experiment_name=moonlight-vexact-reinforce \
    trainer.test_freq=20 \
    trainer.log_val_generations=20 \
    trainer.val_before_train=True \
    trainer.total_epochs=8 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 "${@:1}"
