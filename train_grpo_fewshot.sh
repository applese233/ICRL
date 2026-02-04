#!/bin/bash
# Training script for ICRL GRPO with few-shot web search learning

# Environment setup (modify these paths as needed)
export HUGGINGFACE_HUB_CACHE="~/.cache/huggingface"
export HF_HOME="~/.cache/huggingface"
export RAY_TMPDIR="/tmp/ray"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_DIR='data/nq_search_fewshot'

WAND_PROJECT='ICRL-GRPO'

# Model configuration
export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
export EXPERIMENT_NAME=icrl-grpo-qwen2.5-7B
# export BASE_MODEL='Qwen/Qwen2.5-14B-Instruct'
# export EXPERIMENT_NAME=icrl-grpo-qwen2.5-14B
# export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
# export EXPERIMENT_NAME=icrl-grpo-qwen2.5-3B

export VLLM_ATTENTION_BACKEND=XFORMERS

MAX_PROMPT_LENGTH=5000
MAX_RESPONSE_LENGTH=2048
MAX_START_LENGTH=4096
MAX_OBS_LENGTH=500

TRAIN_BATCH_SIZE=64
PPO_MINI_BATCH_SIZE=32
PPO_MICRO_BATCH_SIZE=8

REWARD_TYPE='fewshot'  # Options: 'fewshot', 'em' (exact match only)
ACCURACY_WEIGHT=0.8
FORMAT_WEIGHT=0.2

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_fewshot \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=128 \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=32 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.max_start_length=$MAX_START_LENGTH \
    data.max_obs_length=$MAX_OBS_LENGTH \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=4 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=10 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=3 \
    trainer.total_training_steps=201 \
    trainer.default_local_dir=./checkpoints/$EXPERIMENT_NAME \
    max_turns=6 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    +reward.type=$REWARD_TYPE \
    +reward.accuracy_weight=$ACCURACY_WEIGHT \
    +reward.format_weight=$FORMAT_WEIGHT \
    2>&1 | tee $EXPERIMENT_NAME.log
