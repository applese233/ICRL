#!/bin/bash
# Advanced Batch Evaluation Script for Search-R1 Checkpoints with vLLM
#
# This script provides fast evaluation using vLLM's efficient inference
#
# Usage:
#   ./eval_batch_vllm.sh --checkpoint /path/to/checkpoint --datasets "triviaqa popqa"
#   ./eval_batch_vllm.sh --help

set -e

# ============================================================================
# Default Configuration
# ============================================================================
export HUGGINGFACE_HUB_CACHE="~/.cache/huggingface"
export HF_HOME="~/.cache/huggingface"

# Default values
CHECKPOINT=""
CHECKPOINT_DIR="./checkpoints"
CHECKPOINT_STEP="global_step_50"
OUTPUT_BASE_DIR="./eval_results"
DATA_DIR="./data/eval"
SEARCH_URL="http://127.0.0.1:8000/retrieve"
DATASETS="triviaqa"
MAX_TURNS=6
MAX_TOKENS=512
TEMPERATURE=0.1
TOPK=3
BATCH_SIZE=16
TENSOR_PARALLEL=1
GPU_MEMORY_UTIL=0.8
NUM_SAMPLES=""
NO_SEARCH=false
USE_VLLM=false
CUDA_DEVICES="0"
CONVERT_ONLY=false
EVAL_ONLY=false

# ============================================================================
# Help Function
# ============================================================================
show_help() {
    echo "Search-R1 Batch Evaluation Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --checkpoint PATH      Path to model checkpoint (required if not using --checkpoint_name)"
    echo "  --checkpoint_name NAME Name of checkpoint in $CHECKPOINT_DIR"
    echo "  --checkpoint_step STEP Checkpoint step (default: $CHECKPOINT_STEP)"
    echo "  --output_dir DIR       Output directory (default: $OUTPUT_BASE_DIR)"
    echo "  --data_dir DIR         Data directory (default: $DATA_DIR)"
    echo "  --datasets LIST        Space-separated list of datasets (default: $DATASETS)"
    echo "                         Available: triviaqa, popqa, hotpotqa, musique, nq, 2wikimultihopqa, bamboogle"
    echo "  --search_url URL       Search API URL (default: $SEARCH_URL)"
    echo "  --max_turns N          Maximum search turns (default: $MAX_TURNS)"
    echo "  --max_tokens N         Maximum tokens per turn (default: $MAX_TOKENS)"
    echo "  --temperature T        Sampling temperature (default: $TEMPERATURE)"
    echo "  --topk K               Number of search results (default: $TOPK)"
    echo "  --batch_size N         Batch size for vLLM (default: $BATCH_SIZE)"
    echo "  --tensor_parallel N    Tensor parallel size (default: $TENSOR_PARALLEL)"
    echo "  --gpu_memory_util F    GPU memory utilization (default: $GPU_MEMORY_UTIL)"
    echo "  --num_samples N        Number of samples to evaluate (default: all)"
    echo "  --no_search            Disable search (direct generation only)"
    echo "  --use_vllm             Use vLLM for faster inference"
    echo "  --cuda_devices DEVICES CUDA visible devices (default: $CUDA_DEVICES)"
    echo "  --convert_only         Only convert datasets, don't evaluate"
    echo "  --eval_only            Skip dataset conversion"
    echo "  --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Evaluate with standard inference"
    echo "  $0 --checkpoint /path/to/model --datasets \"triviaqa popqa\""
    echo ""
    echo "  # Evaluate with vLLM (faster)"
    echo "  $0 --checkpoint /path/to/model --datasets \"triviaqa popqa hotpotqa\" --use_vllm"
    echo ""
    echo "  # Evaluate without search"
    echo "  $0 --checkpoint /path/to/model --datasets triviaqa --no_search"
    echo ""
    echo "  # Only convert datasets"
    echo "  $0 --datasets \"triviaqa popqa hotpotqa musique\" --convert_only"
}

# ============================================================================
# Parse Arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --checkpoint_name)
            CHECKPOINT_NAME="$2"
            shift 2
            ;;
        --checkpoint_step)
            CHECKPOINT_STEP="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --search_url)
            SEARCH_URL="$2"
            shift 2
            ;;
        --max_turns)
            MAX_TURNS="$2"
            shift 2
            ;;
        --max_tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --topk)
            TOPK="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --tensor_parallel)
            TENSOR_PARALLEL="$2"
            shift 2
            ;;
        --gpu_memory_util)
            GPU_MEMORY_UTIL="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --no_search)
            NO_SEARCH=true
            shift
            ;;
        --use_vllm)
            USE_VLLM=true
            shift
            ;;
        --cuda_devices)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --convert_only)
            CONVERT_ONLY=true
            shift
            ;;
        --eval_only)
            EVAL_ONLY=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# ============================================================================
# Setup
# ============================================================================
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build checkpoint path
if [ -n "$CHECKPOINT_NAME" ]; then
    CHECKPOINT="$CHECKPOINT_DIR/$CHECKPOINT_NAME/actor/$CHECKPOINT_STEP"
fi

if [ -z "$CHECKPOINT" ] && [ "$CONVERT_ONLY" = false ]; then
    echo "Error: --checkpoint or --checkpoint_name is required"
    show_help
    exit 1
fi

# Set output directory
if [ -n "$CHECKPOINT" ]; then
    OUTPUT_DIR="$OUTPUT_BASE_DIR/$(basename $(dirname $(dirname $CHECKPOINT)))_$(basename $CHECKPOINT)"
else
    OUTPUT_DIR="$OUTPUT_BASE_DIR/converted_data"
fi

echo "============================================================"
echo "Search-R1 Batch Evaluation"
echo "============================================================"
echo "Checkpoint: $CHECKPOINT"
echo "Output Dir: $OUTPUT_DIR"
echo "Data Dir: $DATA_DIR"
echo "Datasets: $DATASETS"
echo "Use vLLM: $USE_VLLM"
echo "No Search: $NO_SEARCH"
echo "CUDA Devices: $CUDA_DEVICES"
echo "============================================================"

# Create directories
mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR"

# ============================================================================
# Dataset Conversion
# ============================================================================
if [ "$EVAL_ONLY" = false ]; then
    echo ""
    echo ">>> Converting datasets..."
    
    for dataset in $DATASETS; do
        output_file="${DATA_DIR}/${dataset}_eval.parquet"
        
        if [ -f "$output_file" ]; then
            echo "  $dataset: Already exists"
        else
            echo "  $dataset: Converting..."
            python3 "${SCRIPT_DIR}/scripts/eval/convert_datasets.py" \
                --dataset "$dataset" \
                --output_dir "$DATA_DIR" \
                --split validation 2>&1 | sed 's/^/    /'
        fi
    done
fi

if [ "$CONVERT_ONLY" = true ]; then
    echo ""
    echo "Dataset conversion complete. Exiting."
    exit 0
fi

# ============================================================================
# Check Search Server (if search enabled)
# ============================================================================
if [ "$NO_SEARCH" = false ]; then
    echo ""
    echo ">>> Checking search server..."
    
    response=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$SEARCH_URL" \
        -H "Content-Type: application/json" \
        -d '{"queries": ["test"], "topk": 1}' 2>/dev/null || echo "000")
    
    if [ "$response" = "200" ]; then
        echo "  Search server is running."
    else
        echo "  WARNING: Search server not responding (HTTP $response)"
        echo "  Continuing with --no_search mode"
        NO_SEARCH=true
    fi
fi

# ============================================================================
# Run Evaluation
# ============================================================================
echo ""
echo ">>> Running evaluation..."

for dataset in $DATASETS; do
    data_file="${DATA_DIR}/${dataset}_eval.parquet"
    result_dir="${OUTPUT_DIR}/${dataset}"
    
    if [ ! -f "$data_file" ]; then
        echo "  ERROR: Data file not found: $data_file"
        continue
    fi
    
    echo ""
    echo "  Evaluating: $dataset"
    
    # Build command
    if [ "$USE_VLLM" = true ]; then
        eval_script="${SCRIPT_DIR}/scripts/eval/batch_evaluate_vllm.py"
        cmd="python3 $eval_script \
            --checkpoint $CHECKPOINT \
            --data_file $data_file \
            --output_dir $result_dir \
            --search_url $SEARCH_URL \
            --max_turns $MAX_TURNS \
            --max_tokens $MAX_TOKENS \
            --temperature $TEMPERATURE \
            --topk $TOPK \
            --batch_size $BATCH_SIZE \
            --tensor_parallel_size $TENSOR_PARALLEL \
            --gpu_memory_utilization $GPU_MEMORY_UTIL"
    else
        eval_script="${SCRIPT_DIR}/scripts/eval/batch_evaluate.py"
        cmd="python3 $eval_script \
            --checkpoint $CHECKPOINT \
            --data_file $data_file \
            --output_dir $result_dir \
            --search_url $SEARCH_URL \
            --max_turns $MAX_TURNS \
            --max_new_tokens $MAX_TOKENS \
            --temperature $TEMPERATURE \
            --topk $TOPK"
    fi
    
    if [ "$NO_SEARCH" = true ]; then
        cmd="$cmd --no_search"
    fi
    
    if [ -n "$NUM_SAMPLES" ]; then
        cmd="$cmd --num_samples $NUM_SAMPLES"
    fi
    
    # Run evaluation
    echo "  Command: $cmd"
    $cmd 2>&1 | sed 's/^/    /'
done

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================"
echo "Evaluation Complete"
echo "============================================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""

# Print summary
echo "Summary:"
echo "--------"

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
    print(f\"  Samples: {m.get('num_samples', 'N/A')}\")
    print(f\"  EM Accuracy: {m.get('em_accuracy', 0):.4f}\")
    print(f\"  SubEM Accuracy: {m.get('subem_accuracy', 0):.4f}\")
    print(f\"  Avg F1: {m.get('avg_f1', 0):.4f}\")
    print(f\"  Avg Search Count: {m.get('avg_search_count', 0):.2f}\")
" 2>/dev/null || echo "  (Results not available)"
    fi
done

echo ""
echo "Done!"
