set -x

# Model: Qwen3-1.7B
model_path="${model_dir}"

# Data: DAPO-Math-17k (train) / AIME 2024 (val)
train_path="${train_path}"
test_path="${test_path}"

# Register vexact rollout globally
export VERL_USE_EXTERNAL_MODULES=vexact.integrations.verl.register
export VERL_LOGGING_LEVEL=DEBUG
export NCCL_DEBUG=ERROR
# VeOmni Liger Patch
export VEOMNI_USE_LIGER_KERNEL=0
# Use triton-invariant on A100.
vexact_attn_implementation=${INFER_FA_IMPL:-fa-invariant}
vexact_max_cache_blocks=${VEXACT_MAX_CACHE_BLOCKS:-1024}
# Register and enable actor and ref model FSDP ops
verl_model_external_lib=vexact.integrations.verl.fsdp_enable_invariant
# Use triton-invariant on A100.
veomni_attn_implementation=${VEOMNI_ATTN_IMPLEMENTATION:-"flash_attention_3"}
moe_implementation=${VEOMNI_MOE_IMPLEMENTATION:-"fused"}
enforce_eager=${ENFORCE_EAGER:-False}

profile_save_path=./verl_rollout_profile_gsm8k
# Enable fused lce for both training and inference
use_fused_kernels=True
fused_kernel_backend=torch
# Use liger RMSNorm/RoPE/Swiglu in VeRL
use_liger=False

FSDP_SIZE=${FSDP_SIZE:-8}
SP_SIZE=${SP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}


python3 -m verl.trainer.main_ppo \
    model_engine=veomni \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_path/dapo-math-17k.parquet \
    data.val_files=$test_path/aime-2024.parquet \
    data.return_raw_chat=False \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.trust_remote_code=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=$veomni_attn_implementation \
    actor_rollout_ref.model.external_lib=$verl_model_external_lib \
    actor_rollout_ref.model.use_fused_kernels=$use_fused_kernels \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=$fused_kernel_backend \
    actor_rollout_ref.model.use_liger=$use_liger \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.veomni.param_offload=True \
    actor_rollout_ref.actor.veomni.optimizer_offload=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.veomni.fsdp_size="${FSDP_SIZE}" \
    actor_rollout_ref.actor.veomni.ulysses_parallel_size="${SP_SIZE}" \
    actor_rollout_ref.actor.veomni.expert_parallel_size="${EP_SIZE}" \
    actor_rollout_ref.actor.veomni.attn_implementation=$veomni_attn_implementation \
    actor_rollout_ref.actor.veomni.moe_implementation=$moe_implementation \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.veomni.param_offload=True \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vexact \
    ++actor_rollout_ref.rollout.engine_kwargs.vexact.max_cache_blocks=$vexact_max_cache_blocks \
    ++actor_rollout_ref.rollout.engine_kwargs.vexact.attn_impl=$vexact_attn_implementation \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=$enforce_eager \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.layered_summon=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.profiler.enable=True \
    actor_rollout_ref.rollout.profiler.save_path=$profile_save_path \
    actor_rollout_ref.ref.veomni.optimizer_offload=True \
    algorithm.use_kl_in_reward=False \
    reward.reward_manager.name=dapo \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='vexact_test' \
    trainer.experiment_name="vexact_veomni_qwen3_1b7" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 $@ 2>&1 | tee qwen3_1b7.log
