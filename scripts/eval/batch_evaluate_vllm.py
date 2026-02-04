#!/usr/bin/env python3
"""
Fast batch evaluation using vLLM for Search-R1 checkpoints.

This script provides faster evaluation than the standard batch_evaluate.py
by using vLLM's efficient batched inference.

Usage:
    python batch_evaluate_vllm.py \
        --checkpoint /path/to/checkpoint \
        --data_file ./data/eval/triviaqa_eval.parquet \
        --output_dir ./results/triviaqa
"""

import os
import sys
import json
import argparse
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import pandas as pd
import requests
import re
import string

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available. Please install with: pip install vllm")

from transformers import AutoTokenizer


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
    max_tokens: int = 512
    temperature: float = 0.1
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.8
    no_search: bool = False
    num_samples: Optional[int] = None
    batch_size: int = 16
    save_generations: bool = True


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
    return matches[-1].group(1).strip()


def extract_search_query(text: str) -> Optional[str]:
    """Extract search query from <search>...</search> tags."""
    matches = list(re.finditer(r'<search>(.*?)</search>', text, re.DOTALL | re.IGNORECASE))
    if not matches:
        return None
    return matches[-1].group(1).strip()


class VLLMBatchEvaluator:
    """Fast batch evaluator using vLLM."""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.tokenizer = None
        self.llm = None
        
        self._load_model()
        
    def _load_model(self):
        """Load model using vLLM."""
        print(f"Loading model from {self.config.checkpoint}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.checkpoint,
            trust_remote_code=True
        )
        
        self.llm = LLM(
            model=self.config.checkpoint,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=8192,
        )
        
        print("Model loaded successfully")
        
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
    
    def batch_search(self, queries: List[str]) -> List[str]:
        """Batch search for multiple queries."""
        if self.config.no_search:
            return ["[Search disabled]"] * len(queries)
        
        try:
            payload = {
                "queries": queries,
                "topk": self.config.topk,
                "return_scores": True
            }
            response = requests.post(self.config.search_url, json=payload, timeout=60)
            
            if response.status_code != 200:
                return [f"[Search error: status {response.status_code}]"] * len(queries)
            
            all_results = response.json()['result']
            
            formatted_results = []
            for results in all_results:
                formatted = ""
                for idx, doc in enumerate(results):
                    content = doc['document']['contents']
                    title = content.split("\n")[0]
                    text = "\n".join(content.split("\n")[1:])
                    formatted += f"Doc {idx+1}(Title: {title}) {text}\n"
                formatted_results.append(formatted if formatted else "[No results found]")
            
            return formatted_results
            
        except Exception as e:
            return [f"[Search error: {e}]"] * len(queries)
    
    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate responses for a batch of prompts."""
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stop=["</search>", "</answer>"],
            include_stop_str_in_output=True,
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        return [output.outputs[0].text for output in outputs]
    
    def evaluate_with_search(self, samples: List[Dict]) -> List[Dict]:
        """Evaluate samples with multi-turn search."""
        results = []
        
        # Prepare initial prompts
        active_samples = []
        for idx, sample in enumerate(samples):
            prompt_data = sample.get('prompt', [])
            if hasattr(prompt_data, 'tolist'):
                prompt_data = prompt_data.tolist()
            
            if prompt_data and isinstance(prompt_data[0], dict):
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
            
            active_samples.append({
                'idx': idx,
                'sample': sample,
                'prompt': prompt,
                'full_response': '',
                'turn': 0,
                'done': False
            })
        
        # Multi-turn generation loop
        for turn in range(self.config.max_turns):
            # Filter active samples
            pending = [s for s in active_samples if not s['done']]
            
            if not pending:
                break
            
            print(f"  Turn {turn + 1}: {len(pending)} samples active")
            
            # Generate for all pending samples
            prompts = [s['prompt'] + s['full_response'] for s in pending]
            generations = self.generate_batch(prompts)
            
            # Process generations
            search_needed = []
            search_queries = []
            
            for i, gen in enumerate(generations):
                pending[i]['full_response'] += gen
                
                # Check if done
                if extract_answer(pending[i]['full_response']):
                    pending[i]['done'] = True
                elif '</search>' in gen and not self.config.no_search:
                    query = extract_search_query(gen)
                    if query:
                        search_needed.append(i)
                        search_queries.append(query)
                else:
                    pending[i]['done'] = True
            
            # Batch search
            if search_queries:
                search_results = self.batch_search(search_queries)
                
                for i, result in zip(search_needed, search_results):
                    pending[i]['full_response'] += f"\n<information>\n{result}\n</information>\n"
        
        # Compute metrics for all samples
        for s in active_samples:
            sample = s['sample']
            full_response = s['full_response']
            
            # Get golden answers
            golden_answers = sample.get('golden_answers', [])
            if hasattr(golden_answers, 'tolist'):
                golden_answers = golden_answers.tolist()
            
            reward_model = sample.get('reward_model', {})
            if isinstance(reward_model, dict) and 'ground_truth' in reward_model:
                gt = reward_model['ground_truth']
                if isinstance(gt, dict) and 'target' in gt:
                    target = gt['target']
                    if hasattr(target, 'tolist'):
                        golden_answers = target.tolist()
                    else:
                        golden_answers = list(target) if isinstance(target, (list, tuple)) else [target]
            
            # Extract answer and compute metrics
            extracted_answer = extract_answer(full_response) or ""
            
            em = em_check(extracted_answer, golden_answers) if extracted_answer else False
            subem = subem_check(extracted_answer, golden_answers) if extracted_answer else False
            f1 = f1_score(extracted_answer, golden_answers) if extracted_answer else 0.0
            
            search_count = len(re.findall(r'<search>', full_response, re.IGNORECASE))
            think_count = len(re.findall(r'<think>', full_response, re.IGNORECASE))
            
            result = {
                'idx': s['idx'],
                'question': sample.get('question', ''),
                'golden_answers': golden_answers,
                'predicted_answer': extracted_answer,
                'em': em,
                'subem': subem,
                'f1': f1,
                'search_count': search_count,
                'think_count': think_count,
                'data_source': sample.get('data_source', 'unknown'),
            }
            
            if self.config.save_generations:
                result['full_response'] = full_response[:5000]
            
            results.append(result)
        
        return results
    
    def evaluate_dataset(self, data_file: str) -> Dict[str, Any]:
        """Evaluate on a dataset file."""
        print(f"\nEvaluating on {data_file}...")
        
        df = pd.read_parquet(data_file)
        
        if self.config.num_samples:
            df = df.head(self.config.num_samples)
        
        print(f"Loaded {len(df)} samples")
        
        # Convert to list of dicts
        samples = [df.iloc[i].to_dict() for i in range(len(df))]
        
        # Process in batches
        all_results = []
        batch_size = self.config.batch_size
        
        for i in tqdm(range(0, len(samples), batch_size), desc="Evaluating"):
            batch = samples[i:i + batch_size]
            batch_results = self.evaluate_with_search(batch)
            all_results.extend(batch_results)
        
        # Compute metrics
        em_scores = [r['em'] for r in all_results]
        subem_scores = [r['subem'] for r in all_results]
        f1_scores = [r['f1'] for r in all_results]
        search_counts = [r['search_count'] for r in all_results]
        think_counts = [r['think_count'] for r in all_results]
        
        metrics = {
            'dataset': os.path.basename(data_file),
            'num_samples': len(all_results),
            'em_accuracy': np.mean(em_scores),
            'subem_accuracy': np.mean(subem_scores),
            'avg_f1': np.mean(f1_scores),
            'avg_search_count': np.mean(search_counts),
            'avg_think_count': np.mean(think_counts),
        }
        
        return {
            'metrics': metrics,
            'results': all_results
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
        
        all_metrics = []
        for data_file in data_files:
            start_time = time.time()
            eval_result = self.evaluate_dataset(data_file)
            eval_time = time.time() - start_time
            
            eval_result['metrics']['eval_time'] = eval_time
            
            # Save results
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


def main():
    if not VLLM_AVAILABLE:
        print("Error: vLLM is required for this script.")
        print("Please install with: pip install vllm")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Fast batch evaluation using vLLM")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    parser.add_argument('--search_url', type=str, default='http://127.0.0.1:8000/retrieve')
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--max_turns', type=int, default=6)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8)
    parser.add_argument('--no_search', action='store_true')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--no_save_generations', action='store_true')
    
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
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        no_search=args.no_search,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        save_generations=not args.no_save_generations,
    )
    
    evaluator = VLLMBatchEvaluator(config)
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()
