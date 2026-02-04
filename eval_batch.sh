#!/bin/bash
# Batch Evaluation Script for Search-R1 Checkpoints
# This script evaluates trained model checkpoints on various QA datasets
#
# Usage:
#   ./eval_batch.sh [checkpoint_step]
#
# Example:
#   ./eval_batch.sh global_step_50    # Evaluate checkpoint at step 50
#   ./eval_batch.sh                    # Use default checkpoint

# ============================================================================
# Environment Setup
# ============================================================================
export HUGGINGFACE_HUB_CACHE="~/.cache/huggingface"
export HF_HOME="~/.cache/huggingface"

export CUDA_VISIBLE_DEVICES=0

# ============================================================================
# Checkpoint Configuration
# ============================================================================
# Default checkpoint directory and step
CHECKPOINT_DIR="./checkpoints/icrl-grpo-qwen2.5-7B/actor"
CHECKPOINT_STEP="${1:-global_step_50}"
CHECKPOINT_PATH="$CHECKPOINT_DIR/$CHECKPOINT_STEP"

# Alternative checkpoints (uncomment to use):
# CHECKPOINT_PATH="Qwen/Qwen2.5-7B-Instruct"  # Base model
# CHECKPOINT_PATH="your_huggingface_model"     # HuggingFace model

# ============================================================================
# Evaluation Configuration
# ============================================================================
OUTPUT_DIR="./eval_results/$(basename $CHECKPOINT_DIR)_${CHECKPOINT_STEP}"
DATA_DIR="./data/eval"
SEARCH_URL="http://127.0.0.1:8000/retrieve"

# Evaluation parameters
MAX_TURNS=6
MAX_NEW_TOKENS=512
TEMPERATURE=0.1
TOPK=3

# Number of samples to evaluate (set to null for all samples)
NUM_SAMPLES=${NUM_SAMPLES:-null}

# Datasets to evaluate (space-separated)
# Available: triviaqa, popqa, hotpotqa, musique, nq, 2wikimultihopqa, bamboogle
DATASETS="${DATASETS:-triviaqa popqa hotpotqa}"

# ============================================================================
# Script Directory
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/scripts/eval/batch_evaluate.py"
CONVERT_SCRIPT="${SCRIPT_DIR}/scripts/eval/convert_datasets.py"

# ============================================================================
# Functions
# ============================================================================

print_header() {
    echo ""
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

check_checkpoint() {
    if [ ! -d "$CHECKPOINT_PATH" ] && [ ! -f "$CHECKPOINT_PATH/config.json" ]; then
        echo "WARNING: Checkpoint path does not exist or may be a HuggingFace model ID"
        echo "Checkpoint: $CHECKPOINT_PATH"
        # Don't exit - it might be a HuggingFace model ID
    else
        echo "Using checkpoint: $CHECKPOINT_PATH"
    fi
}

check_search_server() {
    echo "Checking search server at $SEARCH_URL..."
    response=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$SEARCH_URL" \
        -H "Content-Type: application/json" \
        -d '{"queries": ["test"], "topk": 1}' 2>/dev/null || echo "000")
    
    if [ "$response" = "200" ]; then
        echo "Search server is running."
        return 0
    else
        echo "WARNING: Search server is not responding (HTTP $response)"
        echo "You may need to start the retrieval server first:"
        echo "  python -m search_r1.search.retrieval_server"
        echo ""
        echo "Do you want to continue with --no_search mode? (y/n)"
        read -r answer
        if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
            NO_SEARCH="--no_search"
            return 0
        else
            echo "Exiting. Please start the search server and try again."
            exit 1
        fi
    fi
}

convert_dataset() {
    local dataset=$1
    local output_file="${DATA_DIR}/${dataset}_eval.parquet"
    
    if [ -f "$output_file" ]; then
        echo "Dataset $dataset already converted: $output_file"
        return 0
    fi
    
    echo "Converting dataset: $dataset"
    python3 "$CONVERT_SCRIPT" \
        --dataset "$dataset" \
        --output_dir "$DATA_DIR" \
        --split validation
    
    return $?
}

evaluate_dataset() {
    local dataset=$1
    local data_file="${DATA_DIR}/${dataset}_eval.parquet"
    local result_dir="${OUTPUT_DIR}/${dataset}"
    
    if [ ! -f "$data_file" ]; then
        echo "ERROR: Data file not found: $data_file"
        echo "Please run dataset conversion first."
        return 1
    fi
    
    echo "Evaluating on $dataset..."
    
    # Build evaluation command
    eval_cmd="python3 $EVAL_SCRIPT \
        --checkpoint $CHECKPOINT_PATH \
        --data_file $data_file \
        --output_dir $result_dir \
        --search_url $SEARCH_URL \
        --max_turns $MAX_TURNS \
        --max_new_tokens $MAX_NEW_TOKENS \
        --temperature $TEMPERATURE \
        --topk $TOPK \
        $NO_SEARCH"
    
    if [ "$NUM_SAMPLES" != "null" ] && [ -n "$NUM_SAMPLES" ]; then
        eval_cmd="$eval_cmd --num_samples $NUM_SAMPLES"
    fi
    
    if [ "$VERBOSE" = "true" ]; then
        eval_cmd="$eval_cmd --verbose"
    fi
    
    echo "Running: $eval_cmd"
    $eval_cmd
    
    return $?
}

# ============================================================================
# Main
# ============================================================================

print_header "Search-R1 Batch Evaluation"

echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Datasets: $DATASETS"
echo "  Search URL: $SEARCH_URL"
echo "  Max Turns: $MAX_TURNS"
echo "  Temperature: $TEMPERATURE"
echo ""

# Check prerequisites
check_checkpoint
NO_SEARCH=""
check_search_server

# Create directories
mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR"

# Convert and evaluate each dataset
print_header "Dataset Conversion"

for dataset in $DATASETS; do
    convert_dataset "$dataset"
done

print_header "Running Evaluation"

TOTAL_DATASETS=$(echo $DATASETS | wc -w)
COMPLETED=0
FAILED=0

for dataset in $DATASETS; do
    echo ""
    echo ">>> Evaluating $dataset ($((COMPLETED+1))/$TOTAL_DATASETS)"
    
    if evaluate_dataset "$dataset"; then
        COMPLETED=$((COMPLETED+1))
    else
        FAILED=$((FAILED+1))
        echo "WARNING: Evaluation failed for $dataset"
    fi
done

# ============================================================================
# Summary
# ============================================================================
print_header "Evaluation Complete"

echo "Results saved to: $OUTPUT_DIR"
echo "Completed: $COMPLETED/$TOTAL_DATASETS"
if [ $FAILED -gt 0 ]; then
    echo "Failed: $FAILED"
fi

# Print summary from result files
echo ""
echo "Summary of Results:"
echo "-------------------"

for dataset in $DATASETS; do
    result_file="${OUTPUT_DIR}/${dataset}/${dataset}_eval_results.json"
    if [ -f "$result_file" ]; then
        echo ""
        echo "[$dataset]"
        python3 -c "
import json
with open('$result_file') as f:
    data = json.load(f)
    m = data.get('metrics', {})
    print(f\"  EM Accuracy: {m.get('em_accuracy', 0):.4f}\")
    print(f\"  SubEM Accuracy: {m.get('subem_accuracy', 0):.4f}\")
    print(f\"  Avg F1: {m.get('avg_f1', 0):.4f}\")
    print(f\"  Avg Search Count: {m.get('avg_search_count', 0):.2f}\")
" 2>/dev/null || echo "  (Results not available)"
    fi
done

echo ""
echo "Done!"
