#!/bin/bash
set -x

# Model: Qwen3-30B-A3B-Base
model_path=/mnt/hdfs/model_path

# Data: DAPO-Math-17k (train) / AIME 2024 (val)
data_train_path=/mnt/hdfs/data_train_path
data_test_path=/mnt/hdfs/data_test_path


# Register vexact rollout globally
export VERL_USE_EXTERNAL_MODULES=vexact.integrations.verl.register
export VERL_LOGGING_LEVEL=DEBUG
export NCCL_DEBUG=ERROR
# VeOmni Liger Patch
export VEOMNI_USE_LIGER_KERNEL=0
# H100 (SM90) uses FA3 kernel for batch-invariant inference
export INFER_FA_IMPL=${INFER_FA_IMPL:-fa-invariant}
# Register and enable actor and ref model FSDP ops
verl_model_external_lib=vexact.integrations.verl.fsdp_enable_invariant
attn_implementation=flash_attention_3
# Enable fused lce for both training and inference
use_fused_kernels=True
fused_kernel_backend=torch
# Use liger RMSNorm/RoPE/Swiglu
use_liger=False



echo "${@:1}"

loss_agg_mode="seq-mean-token-sum-norm"

# Checkpoint
exp_name="${EXP_NAME:-qwen3_moe_vexact_reinforce_$(date +%Y%m%d)}"
hdfs_base="/mnt/hdfs/vexact_exp/${exp_name}"

checkpoint_path="${hdfs_base}/checkpoints"


RAY_DEDUP_LOGS=0 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
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
    actor_rollout_ref.model.path=$model_path \
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
    actor_rollout_ref.actor.veomni.fsdp_size=64 \
    actor_rollout_ref.actor.veomni.expert_parallel_size=1 \
    actor_rollout_ref.actor.veomni.attn_implementation=$attn_implementation \
    actor_rollout_ref.actor.veomni.param_offload=True \
    actor_rollout_ref.actor.veomni.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vexact \
    ++actor_rollout_ref.rollout.engine_kwargs.vexact.attn_impl=$INFER_FA_IMPL \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=20480 \
    ++actor_rollout_ref.rollout.engine_kwargs.vexact.max_cache_blocks=1024 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    custom_reward_function.path=examples/math_reward_model/math_grader_moe.py \
    custom_reward_function.name=compute_math_score \
    trainer.project_name=vexact-baseline-math-moe-reinforce \
    trainer.experiment_name=vexact-exp-MOE \
    trainer.test_freq=20 \
    trainer.log_val_generations=20 \
    trainer.val_before_train=True \
    trainer.save_freq=50 \
    trainer.default_local_dir=$checkpoint_path \
    trainer.resume_mode=auto \
    trainer.total_epochs=8 \
    trainer.nnodes=8 \
    trainer.n_gpus_per_node=8 "${@:1}"
