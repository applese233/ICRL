#!/usr/bin/env python3
"""
Batch evaluation script for Search-R1 trained checkpoints on various QA datasets.

This script evaluates a trained model checkpoint on QA datasets using:
1. Multi-turn web search capability (via retrieval API)
2. Exact match (EM) and substring match evaluation
3. Optional format compliance scoring

Usage:
    # Evaluate on a single parquet file
    python batch_evaluate.py \
        --checkpoint /path/to/checkpoint \
        --data_file ./data/eval/triviaqa_eval.parquet \
        --output_dir ./results/triviaqa

    # Evaluate on multiple datasets
    python batch_evaluate.py \
        --checkpoint /path/to/checkpoint \
        --data_dir ./data/eval \
        --output_dir ./results

    # Evaluate without search (direct generation)
    python batch_evaluate.py \
        --checkpoint /path/to/checkpoint \
        --data_file ./data/eval/triviaqa_eval.parquet \
        --no_search
"""

import os
import sys
import json
import argparse
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import pandas as pd
import requests
import re
import string


@dataclass
class EvalConfig:
    """Configuration for batch evaluation."""
    checkpoint: str
    data_file: Optional[str] = None
    data_dir: Optional[str] = None
    output_dir: str = "./eval_results"
    search_url: str = "http://127.0.0.1:8000/retrieve"
    topk: int = 3
    max_turns: int = 6
    max_new_tokens: int = 512
    temperature: float = 0.1  # Lower temperature for evaluation
    do_sample: bool = True
    batch_size: int = 1
    no_search: bool = False
    device: str = "auto"
    verbose: bool = False
    save_generations: bool = True
    num_samples: Optional[int] = None


class StopOnSequence(StoppingCriteria):
    """Custom stopping criteria for generation."""
    
    def __init__(self, target_sequences: List[str], tokenizer):
        self.target_ids = [tokenizer.encode(seq, add_special_tokens=False) for seq in target_sequences]
        self.target_lengths = [len(ids) for ids in self.target_ids]
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        for i, target_id in enumerate(self.target_ids):
            if input_ids.shape[1] >= self.target_lengths[i]:
                target = torch.as_tensor(target_id, device=input_ids.device)
                if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                    return True
        return False


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction: str, golden_answers: List[str]) -> bool:
    """Check exact match."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_pred = normalize_answer(prediction)
    for answer in golden_answers:
        if normalize_answer(answer) == normalized_pred:
            return True
    return False


def subem_check(prediction: str, golden_answers: List[str]) -> bool:
    """Check if answer is contained in prediction."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_pred = normalize_answer(prediction)
    for answer in golden_answers:
        if normalize_answer(answer) in normalized_pred:
            return True
    return False


def f1_score(prediction: str, golden_answers: List[str]) -> float:
    """Calculate token-level F1 score."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    
    pred_tokens = set(normalize_answer(prediction).split())
    
    best_f1 = 0.0
    for answer in golden_answers:
        gold_tokens = set(normalize_answer(answer).split())
        
        if not pred_tokens or not gold_tokens:
            continue
            
        common = pred_tokens & gold_tokens
        num_common = len(common)
        
        if num_common == 0:
            continue
            
        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        best_f1 = max(best_f1, f1)
    
    return best_f1


def extract_answer(text: str) -> Optional[str]:
    """Extract answer from <answer>...</answer> tags."""
    matches = list(re.finditer(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE))
    if not matches:
        return None
    # Return the last answer (in case of few-shot examples)
    return matches[-1].group(1).strip()


def extract_search_query(text: str) -> Optional[str]:
    """Extract search query from <search>...</search> tags."""
    matches = list(re.finditer(r'<search>(.*?)</search>', text, re.DOTALL | re.IGNORECASE))
    if not matches:
        return None
    return matches[-1].group(1).strip()


class BatchEvaluator:
    """Batch evaluator for Search-R1 checkpoints."""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        
        # Load model
        self._load_model()
        
        # Setup stopping criteria
        self.stop_sequences = [
            "</search>", " </search>", "</search>\n", 
            "</answer>", " </answer>", "</answer>\n"
        ]
        
    def _load_model(self):
        """Load model and tokenizer from checkpoint."""
        print(f"Loading model from {self.config.checkpoint}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.checkpoint, 
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.checkpoint,
            torch_dtype=torch.bfloat16,
            device_map=self.config.device,
            trust_remote_code=True,
        )
        self.model.eval()
        
        self.device = next(self.model.parameters()).device
        print(f"Model loaded on {self.device}")
        
    def search(self, query: str) -> str:
        """Call search API."""
        if self.config.no_search:
            return "[Search disabled]"
            
        try:
            payload = {
                "queries": [query],
                "topk": self.config.topk,
                "return_scores": True
            }
            response = requests.post(self.config.search_url, json=payload, timeout=30)
            
            if response.status_code != 200:
                return f"[Search error: status {response.status_code}]"
            
            results = response.json()['result'][0]
            
            formatted = ""
            for idx, doc in enumerate(results):
                content = doc['document']['contents']
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                formatted += f"Doc {idx+1}(Title: {title}) {text}\n"
            
            return formatted if formatted else "[No results found]"
            
        except Exception as e:
            return f"[Search error: {e}]"
    
    def generate_response(self, prompt: str) -> Tuple[str, str]:
        """Generate response with multi-turn search.
        
        The model generates text freely, but we post-process to detect </search> and </answer> tags.
        When </search> is found, we call the real search API and continue generation.
        
        Returns:
            Tuple of (full_response, extracted_answer)
        """
        # Track full response
        full_response = ""
        current_prompt = prompt
        
        for turn in range(self.config.max_turns):
            # Tokenize
            inputs = self.tokenizer(
                current_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=8192
            ).to(self.device)
            
            # Generate - let model generate freely, we'll post-process
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode new tokens
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            if self.config.verbose:
                print(f"    [Turn {turn+1} raw generation: {len(new_text)} chars]", flush=True)
            
            # Post-process: Find first </search> or </answer> and truncate
            # This prevents the model's fake <information> from being used
            search_pos = new_text.find('</search>')
            answer_pos = new_text.find('</answer>')
            
            # Determine which comes first
            if search_pos >= 0 and answer_pos >= 0:
                if search_pos < answer_pos:
                    # </search> comes first - truncate there and do real search
                    new_text = new_text[:search_pos + len('</search>')]
                else:
                    # </answer> comes first - truncate there and we're done
                    new_text = new_text[:answer_pos + len('</answer>')]
            elif search_pos >= 0:
                # Only </search> found
                new_text = new_text[:search_pos + len('</search>')]
            elif answer_pos >= 0:
                # Only </answer> found
                new_text = new_text[:answer_pos + len('</answer>')]
            
            full_response += new_text
            
            # Check if answer is provided (we're done)
            if '</answer>' in new_text:
                break
            
            # Check if search is needed
            if '</search>' in new_text and not self.config.no_search:
                search_query = extract_search_query(new_text)
                if search_query:
                    if self.config.verbose:
                        print(f"    [Calling search API: {search_query[:50]}...]", flush=True)
                    search_results = self.search(search_query)
                    search_text = f"\n<information>\n{search_results}\n</information>\n"
                    full_response += search_text
                    current_prompt = prompt + full_response
                    
                    if self.config.verbose:
                        print(f"    [Search results added, continuing generation]", flush=True)
                else:
                    # No valid search query extracted
                    break
            else:
                # No search tag and no answer - model didn't produce expected format
                if self.config.verbose:
                    print(f"    [No </search> or </answer> found in generation]", flush=True)
                break
        
        extracted_answer = extract_answer(full_response) or ""
        return full_response, extracted_answer
    
    def evaluate_sample(self, sample: Dict) -> Dict[str, Any]:
        """Evaluate a single sample."""
        question = sample.get('question', '')
        golden_answers = sample.get('golden_answers', [])
        
        # Handle numpy arrays
        if hasattr(golden_answers, 'tolist'):
            golden_answers = golden_answers.tolist()
        
        # Get ground truth from reward_model if available
        reward_model = sample.get('reward_model', {})
        if isinstance(reward_model, dict) and 'ground_truth' in reward_model:
            gt = reward_model['ground_truth']
            if isinstance(gt, dict) and 'target' in gt:
                target = gt['target']
                if hasattr(target, 'tolist'):
                    golden_answers = target.tolist()
                else:
                    golden_answers = list(target) if isinstance(target, (list, tuple)) else [target]
        
        # Build prompt
        prompt_data = sample.get('prompt', [])
        if hasattr(prompt_data, 'tolist'):
            prompt_data = prompt_data.tolist()
        
        if prompt_data and isinstance(prompt_data[0], dict):
            # Apply chat template
            if self.tokenizer.chat_template:
                prompt = self.tokenizer.apply_chat_template(
                    prompt_data, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                prompt = prompt_data[0].get('content', '')
        else:
            prompt = str(prompt_data)
        
        # Generate response
        start_time = time.time()
        full_response, extracted_answer = self.generate_response(prompt)
        generation_time = time.time() - start_time
        
        # Compute metrics
        em = em_check(extracted_answer, golden_answers) if extracted_answer else False
        subem = subem_check(extracted_answer, golden_answers) if extracted_answer else False
        f1 = f1_score(extracted_answer, golden_answers) if extracted_answer else 0.0
        
        # Count search turns
        search_count = len(re.findall(r'<search>', full_response, re.IGNORECASE))
        think_count = len(re.findall(r'<think>', full_response, re.IGNORECASE))
        
        result = {
            'question': question,
            'golden_answers': golden_answers,
            'predicted_answer': extracted_answer,
            'em': em,
            'subem': subem,
            'f1': f1,
            'search_count': search_count,
            'think_count': think_count,
            'generation_time': generation_time,
        }
        
        if self.config.save_generations:
            result['full_response'] = full_response[:5000]  # Truncate for storage
        
        return result
    
    def evaluate_dataset(self, data_file: str) -> Dict[str, Any]:
        """Evaluate on a dataset file."""
        print(f"\nEvaluating on {data_file}...", flush=True)
        
        # Load data
        df = pd.read_parquet(data_file)
        
        if self.config.num_samples:
            df = df.head(self.config.num_samples)
        
        print(f"Loaded {len(df)} samples", flush=True)
        print(f"Starting evaluation (this may take a while)...", flush=True)
        
        # Evaluate
        results = []
        for idx in tqdm(range(len(df)), desc="Evaluating", file=sys.stdout):
            sample = df.iloc[idx].to_dict()
            result = self.evaluate_sample(sample)
            result['idx'] = idx
            result['data_source'] = sample.get('data_source', 'unknown')
            results.append(result)
            
            if self.config.verbose and idx < 5:
                print(f"\n--- Sample {idx} ---")
                print(f"Question: {result['question']}")
                print(f"Golden: {result['golden_answers']}")
                print(f"Predicted: {result['predicted_answer']}")
                print(f"EM: {result['em']}, SubEM: {result['subem']}, F1: {result['f1']:.3f}")
        
        # Compute aggregate metrics
        em_scores = [r['em'] for r in results]
        subem_scores = [r['subem'] for r in results]
        f1_scores = [r['f1'] for r in results]
        search_counts = [r['search_count'] for r in results]
        think_counts = [r['think_count'] for r in results]
        gen_times = [r['generation_time'] for r in results]
        
        metrics = {
            'dataset': os.path.basename(data_file),
            'num_samples': len(results),
            'em_accuracy': np.mean(em_scores),
            'subem_accuracy': np.mean(subem_scores),
            'avg_f1': np.mean(f1_scores),
            'avg_search_count': np.mean(search_counts),
            'avg_think_count': np.mean(think_counts),
            'avg_generation_time': np.mean(gen_times),
            'total_time': sum(gen_times),
        }
        
        return {
            'metrics': metrics,
            'results': results
        }
    
    def run_evaluation(self):
        """Run evaluation on all specified datasets."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Collect data files
        data_files = []
        if self.config.data_file:
            data_files.append(self.config.data_file)
        elif self.config.data_dir:
            for f in os.listdir(self.config.data_dir):
                if f.endswith('.parquet'):
                    data_files.append(os.path.join(self.config.data_dir, f))
        
        if not data_files:
            print("No data files found!")
            return
        
        print(f"Found {len(data_files)} dataset(s) to evaluate")
        
        # Evaluate each dataset
        all_metrics = []
        for data_file in data_files:
            eval_result = self.evaluate_dataset(data_file)
            
            # Save detailed results
            dataset_name = os.path.splitext(os.path.basename(data_file))[0]
            results_path = os.path.join(self.config.output_dir, f'{dataset_name}_results.json')
            
            with open(results_path, 'w') as f:
                json.dump(eval_result, f, indent=2, default=str)
            
            print(f"\nResults for {dataset_name}:")
            for k, v in eval_result['metrics'].items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
            
            all_metrics.append(eval_result['metrics'])
        
        # Save summary
        summary_path = os.path.join(self.config.output_dir, 'evaluation_summary.json')
        summary = {
            'checkpoint': self.config.checkpoint,
            'config': {
                'max_turns': self.config.max_turns,
                'temperature': self.config.temperature,
                'topk': self.config.topk,
                'no_search': self.config.no_search,
            },
            'datasets': all_metrics
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nEvaluation complete. Summary saved to {summary_path}")
        
        # Print overall summary
        if len(all_metrics) > 1:
            print("\n=== Overall Summary ===")
            avg_em = np.mean([m['em_accuracy'] for m in all_metrics])
            avg_subem = np.mean([m['subem_accuracy'] for m in all_metrics])
            avg_f1 = np.mean([m['avg_f1'] for m in all_metrics])
            print(f"Average EM: {avg_em:.4f}")
            print(f"Average SubEM: {avg_subem:.4f}")
            print(f"Average F1: {avg_f1:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation for Search-R1 checkpoints")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--data_file', type=str, default=None,
                        help='Path to a single parquet file')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing parquet files')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                        help='Output directory for results')
    parser.add_argument('--search_url', type=str, default='http://127.0.0.1:8000/retrieve',
                        help='Search API URL')
    parser.add_argument('--topk', type=int, default=3,
                        help='Number of search results to retrieve')
    parser.add_argument('--max_turns', type=int, default=6,
                        help='Maximum search turns')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum new tokens per turn')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Sampling temperature')
    parser.add_argument('--no_search', action='store_true',
                        help='Disable search (direct generation only)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed output')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (for debugging)')
    parser.add_argument('--no_save_generations', action='store_true',
                        help='Do not save full generations')
    
    args = parser.parse_args()
    
    if not args.data_file and not args.data_dir:
        print("Error: Either --data_file or --data_dir must be specified")
        sys.exit(1)
    
    config = EvalConfig(
        checkpoint=args.checkpoint,
        data_file=args.data_file,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        search_url=args.search_url,
        topk=args.topk,
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        no_search=args.no_search,
        device=args.device,
        verbose=args.verbose,
        num_samples=args.num_samples,
        save_generations=not args.no_save_generations,
    )
    
    evaluator = BatchEvaluator(config)
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()
