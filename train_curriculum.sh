#!/bin/bash
# Progressive Curriculum Training Script
# Automatically trains through 3 stages: 3-shot -> 2-shot -> 0-shot
# Each stage trains for a specified number of steps, then loads checkpoint for next stage

set -e

# Environment setup (modify these paths as needed)
export HUGGINGFACE_HUB_CACHE="~/.cache/huggingface"
export HF_HOME="~/.cache/huggingface"
export RAY_TMPDIR="/tmp/ray"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ATTENTION_BACKEND=XFORMERS

# ==================== Configuration ====================
WAND_PROJECT='ICRL-Curriculum'
BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
CHECKPOINT_DIR='./checkpoints'

# Training steps per stage
STAGE1_STEPS=100  # 3-shot
STAGE2_STEPS=100  # 2-shot  
STAGE3_STEPS=100  # 0-shot

# Save frequency (should divide stage steps evenly for clean checkpoints)
SAVE_FREQ=50

# ==================== Common Training Parameters ====================
MAX_PROMPT_LENGTH=5000
MAX_RESPONSE_LENGTH=2048
MAX_START_LENGTH=4096
MAX_OBS_LENGTH=500
TRAIN_BATCH_SIZE=64
PPO_MINI_BATCH_SIZE=32
PPO_MICRO_BATCH_SIZE=8
ACCURACY_WEIGHT=0.8
FORMAT_WEIGHT=0.2

run_training_stage() {
    local STAGE=$1
    local NUM_EXAMPLES=$2
    local TOTAL_STEPS=$3
    local MODEL_PATH=$4
    local EXPERIMENT_NAME=$5
    local DATA_DIR=$6

    echo "=========================================="
    echo "Starting Stage $STAGE: ${NUM_EXAMPLES}-shot training"
    echo "Model: $MODEL_PATH"
    echo "Data: $DATA_DIR"
    echo "Steps: $TOTAL_STEPS"
    echo "=========================================="

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
        actor_rollout_ref.model.path=$MODEL_PATH \
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
        trainer.save_freq=$SAVE_FREQ \
        trainer.test_freq=10 \
        trainer.project_name=$WAND_PROJECT \
        trainer.experiment_name=$EXPERIMENT_NAME \
        trainer.total_epochs=1 \
        trainer.total_training_steps=$TOTAL_STEPS \
        trainer.default_local_dir=$CHECKPOINT_DIR/$EXPERIMENT_NAME \
        max_turns=6 \
        retriever.url="http://127.0.0.1:8000/retrieve" \
        retriever.topk=3 \
        +reward.type=fewshot \
        +reward.accuracy_weight=$ACCURACY_WEIGHT \
        +reward.format_weight=$FORMAT_WEIGHT \
        2>&1 | tee ${EXPERIMENT_NAME}.log

    echo "Stage $STAGE completed!"
    echo ""
}

# ==================== Stage 1: 3-shot ====================
STAGE1_EXPERIMENT="icrl-stage1-3shot"
run_training_stage 1 3 $STAGE1_STEPS \
    "$BASE_MODEL" \
    "$STAGE1_EXPERIMENT" \
    "data/nq_search_fewshot"  # 3-shot data

# Get the latest checkpoint from stage 1
STAGE1_CKPT="$CHECKPOINT_DIR/$STAGE1_EXPERIMENT/actor/global_step_${STAGE1_STEPS}"
if [ ! -d "$STAGE1_CKPT" ]; then
    # Try to find the closest checkpoint
    STAGE1_CKPT=$(ls -d $CHECKPOINT_DIR/$STAGE1_EXPERIMENT/actor/global_step_* 2>/dev/null | sort -V | tail -1)
fi
echo "Stage 1 checkpoint: $STAGE1_CKPT"

# ==================== Stage 2: 2-shot ====================
STAGE2_EXPERIMENT="icrl-stage2-2shot"
run_training_stage 2 2 $STAGE2_STEPS \
    "$STAGE1_CKPT" \
    "$STAGE2_EXPERIMENT" \
    "data/nq_search_2shot"  # 2-shot data

# Get the latest checkpoint from stage 2
STAGE2_CKPT="$CHECKPOINT_DIR/$STAGE2_EXPERIMENT/actor/global_step_${STAGE2_STEPS}"
if [ ! -d "$STAGE2_CKPT" ]; then
    STAGE2_CKPT=$(ls -d $CHECKPOINT_DIR/$STAGE2_EXPERIMENT/actor/global_step_* 2>/dev/null | sort -V | tail -1)
fi
echo "Stage 2 checkpoint: $STAGE2_CKPT"

# ==================== Stage 3: 0-shot ====================
STAGE3_EXPERIMENT="icrl-stage3-0shot"
run_training_stage 3 0 $STAGE3_STEPS \
    "$STAGE2_CKPT" \
    "$STAGE3_EXPERIMENT" \
    "data/nq_search_0shot"  # 0-shot data

echo "=========================================="
echo "All stages completed!"
echo "Final model: $CHECKPOINT_DIR/$STAGE3_EXPERIMENT/actor/global_step_${STAGE3_STEPS}"
echo "=========================================="
