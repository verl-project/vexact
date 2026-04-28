#!/bin/bash
set -x

export MODELING_BACKEND=hf

# Model: DeepSeek-R1-Distill-Qwen-1.5B
model_path=/mnt/hdfs/model_path

# Data: MATH (math_1460 train) / AIME 2024 + AIME 2025 (val)
data_path=/mnt/hdfs/data_path

echo "${@:1}"

loss_agg_mode="seq-mean-token-sum-norm"

# Train over a single node, 8 A100-80GB GPUs.
RAY_DEDUP_LOGS=0 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$data_path/train/math_1460.parquet \
    data.val_files=[$data_path/test/aime_2024.parquet,$data_path/test/aime_2025.parquet] \
    data.train_batch_size=64 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.seed=42 \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.actor.optim.betas="[0.9,0.95]" \
    +actor_rollout_ref.actor.optim.override_optimizer_config.eps=1e-15 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    custom_reward_function.path=examples/math_reward_model/math_grader.py \
    custom_reward_function.name=compute_math_score \
    trainer.project_name=dense-math \
    trainer.experiment_name=vllm-baseline \
    trainer.use_legacy_worker_impl=disable \
    trainer.test_freq=50 \
    trainer.log_val_generations=20 \
    trainer.val_before_train=True \
    trainer.total_epochs=88 \
    trainer.n_gpus_per_node=8 "${@:1}"
