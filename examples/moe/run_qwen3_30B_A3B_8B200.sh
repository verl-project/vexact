set -x

# Model: Qwen3-30B-A3B
model_path="${model_dir:-/mnt/hdfs/model_path}"

# Data: gsm8k (parquet)
data_path="${data_dir:-/mnt/hdfs/data_path}"

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

FSDP_SIZE=${FSDP_SIZE:-8}
SP_SIZE=${SP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}

# Profiler: set PROFILE_STEPS to a comma-separated list of training steps to profile, e.g. "3,4,5"
# Output traces are saved to PROFILE_SAVE_PATH (default: outputs/profile)
profile_tool=${PROFILE_TOOL:-"torch"}
profile_steps=${PROFILE_STEPS:-"[3,4,5]"}
profile_save_path=${PROFILE_SAVE_PATH:-"./verl_profile"}
profile_contents=${PROFILE_CONTENTS:-"[cuda,cpu]"}

python3 -m verl.trainer.main_ppo \
    model_engine=veomni \
    algorithm.adv_estimator=grpo \
    data.train_files=$data_path/gsm8k/train.parquet \
    data.val_files=$data_path/gsm8k/test.parquet \
    data.return_raw_chat=False \
    data.train_batch_size=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=$attn_implementation \
    actor_rollout_ref.model.external_lib=$verl_model_external_lib \
    actor_rollout_ref.model.use_fused_kernels=$use_fused_kernels \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=$fused_kernel_backend \
    actor_rollout_ref.model.use_liger=$use_liger \
    actor_rollout_ref.actor.veomni.param_offload=True \
    actor_rollout_ref.actor.veomni.optimizer_offload=True \
    actor_rollout_ref.actor.ppo_epochs=2 \
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
    actor_rollout_ref.actor.veomni.attn_implementation=$attn_implementation \
    actor_rollout_ref.actor.veomni.moe_implementation=$moe_implementation \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.veomni.param_offload=True \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vexact \
    ++actor_rollout_ref.rollout.engine_kwargs.vexact.max_cache_blocks=4608 \
    ++actor_rollout_ref.rollout.engine_kwargs.vexact.attn_impl=$INFER_FA_IMPL \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.layered_summon=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.veomni.optimizer_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.use_legacy_worker_impl=disable \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_example_gsm8k_math' \
    trainer.experiment_name='exact_moe_qwen3_30b' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=80 \
    global_profiler.tool=$profile_tool \
    global_profiler.steps=$profile_steps \
    global_profiler.save_path=$profile_save_path \
    actor_rollout_ref.actor.profiler.enable=true \
    actor_rollout_ref.actor.profiler.tool_config.torch.contents=$profile_contents \
    actor_rollout_ref.rollout.profiler.enable=true \
    actor_rollout_ref.rollout.profiler.save_path=$profile_save_path \
    $@ 2>&1 | tee qwen3_30b.log
