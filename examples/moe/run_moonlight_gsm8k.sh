#!/bin/bash

# Fast-debug variant of run_moonlight_reinforce.sh: Moonlight-16B-A3B-Instruct
# on gsm8k with short prompts/responses and small batches so an end-to-end
# iteration (val_before_train + first training step) finishes in minutes.
#
# Example invocation:
#   MODEL_PATH=/path/to/Moonlight-16B-A3B-Instruct \
#   DATA_PATH=/path/to/gsm8k_root \
#   bash examples/moe/run_moonlight_gsm8k.sh

set -x
model_path=${MODEL_PATH}
# gsm8k parquet root containing train.parquet and test.parquet
data_path=${DATA_PATH}

# Register vexact rollout globally
vexact_verl_register_module=vexact.integrations.verl.register
if [[ -n "${VERL_USE_EXTERNAL_MODULES:-}" ]]; then
    if [[ ",${VERL_USE_EXTERNAL_MODULES}," != *",${vexact_verl_register_module},"* ]]; then
        export VERL_USE_EXTERNAL_MODULES="${vexact_verl_register_module},${VERL_USE_EXTERNAL_MODULES}"
    fi
else
    export VERL_USE_EXTERNAL_MODULES="${vexact_verl_register_module}"
fi
export VERL_LOGGING_LEVEL=DEBUG
export NCCL_DEBUG=ERROR
# B200 (SM100+) uses FA4 CUTE kernel for batch-invariant inference
export INFER_FA_IMPL=${INFER_FA_IMPL:-fa-invariant-cute}
# Register and enable actor and ref model FSDP ops
verl_model_external_lib=vexact.integrations.verl.fsdp_enable_invariant
attn_implementation=${VEOMNI_ATTN_IMPL:-veomni_flash_attention_4_with_sp}
moe_implementation=${VEOMNI_MOE_IMPLEMENTATION:-"fused_quack"}
use_fused_kernels=True
fused_kernel_backend=torch
use_liger=False
vexact_max_cache_blocks=${VEXACT_MAX_CACHE_BLOCKS:-1024}
echo "${@:1}"
loss_agg_mode="seq-mean-token-sum-norm"

RAY_DEDUP_LOGS=1 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    model_engine=veomni \
    algorithm.adv_estimator=grpo \
    data.train_files=$data_path/train.parquet \
    data.val_files=$data_path/test.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
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
    actor_rollout_ref.ref.veomni.param_offload=True \
    actor_rollout_ref.ref.veomni.optimizer_offload=True \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vexact \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    ++actor_rollout_ref.rollout.engine_kwargs.vexact.max_cache_blocks=$vexact_max_cache_blocks \
    ++actor_rollout_ref.rollout.engine_kwargs.vexact.attn_impl=$INFER_FA_IMPL \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    algorithm.use_kl_in_reward=False \
    trainer.project_name=vexact-baseline-gsm8k-moe \
    trainer.experiment_name=moonlight-vexact-gsm8k \
    trainer.logger=[console] \
    trainer.test_freq=5 \
    trainer.log_val_generations=10 \
    trainer.val_before_train=True \
    trainer.total_epochs=1 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 "${@:1}"
