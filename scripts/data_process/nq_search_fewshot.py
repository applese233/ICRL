# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the nq dataset to parquet format with few-shot examples for web search learning.
Based on nq_search.py but adds few-shot demonstrations in the prompt.
"""

import os
import re
import datasets
from pathlib import Path

from verl.utils.hdfs_io import copy, makedirs
import argparse


# Default path to few-shot examples file
DEFAULT_EXAMPLES_FILE = Path(__file__).resolve().parents[2] / "example" / "fewshot_examples.txt"


# System prompt for few-shot (with examples)
SYSTEM_PROMPT_FEWSHOT = """Solve the following problem step by step. You must conduct reasoning inside <think> and </think> every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as you want. Finally, you should provide the answer inside <answer> and </answer> without detailed illustrations. For example, <answer> Beijing </answer>.

Each reasoning step should be wrapped with <think> your thought here </think>.

When you call the search tool, the query should be placed inside <search> query text here </search>.

And the result of the search should be wrapped with <information> search results here </information>.

The last part of your response should be in the following format: <answer> The final answer goes here. </answer>

Here are some existing QA examples that you can refer to:
"""

# System prompt for zero-shot (without examples)
SYSTEM_PROMPT_ZEROSHOT = """Solve the following problem step by step. You must conduct reasoning inside <think> and </think> every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as you want. Finally, you should provide the answer inside <answer> and </answer> without detailed illustrations. For example, <answer> Beijing </answer>.

Each reasoning step should be wrapped with <think> your thought here </think>.

When you call the search tool, the query should be placed inside <search> query text here </search>.

And the result of the search should be wrapped with <information> search results here </information>.

The last part of your response should be in the following format: <answer> The final answer goes here. </answer>
"""

# For backward compatibility
SYSTEM_PROMPT = SYSTEM_PROMPT_FEWSHOT

PROBLEM_TEXT = "Now solve the following problem with the ability you just learned from the examples. Remember, DO NOT take the problem from examples as the problem you need to solve."


def load_fewshot_examples(examples_file: str = None, num_examples: int = None) -> str:
    """Load few-shot examples from file.
    
    Args:
        examples_file: Path to the examples file
        num_examples: Number of examples to include (None for all, or 1, 2, 3, etc.)
    
    Returns:
        String containing the selected examples
    """
    if examples_file is None:
        examples_file = DEFAULT_EXAMPLES_FILE
    else:
        examples_file = Path(examples_file)
    
    if not examples_file.exists():
        print(f"Warning: Few-shot examples file not found at {examples_file}")
        return ""
    
    full_text = examples_file.read_text(encoding="utf-8").strip()
    
    # If num_examples is None, return all examples
    if num_examples is None:
        return full_text
    
    # Parse and extract individual examples
    # Examples are separated by "===============Example N==============="
    example_pattern = r'(===============Example \d+===============.*?)(?================Example \d+===============|$)'
    examples = re.findall(example_pattern, full_text, re.DOTALL)
    
    if not examples:
        # Fallback: return full text if pattern doesn't match
        print("Warning: Could not parse examples, returning full text")
        return full_text
    
    # Select the requested number of examples
    selected_examples = examples[:num_examples]
    
    return "\n".join([ex.strip() for ex in selected_examples])


def build_fewshot_prompt(question: str, fewshot_examples: str) -> str:
    """Build the full prompt with system prompt, few-shot examples, and the actual question."""
    parts = [SYSTEM_PROMPT_FEWSHOT]
    if fewshot_examples:
        parts.append(fewshot_examples)
    parts.append(PROBLEM_TEXT)
    parts.append(f"Actual Problem: {question}")
    return "\n\n".join(parts)


def build_zeroshot_prompt(question: str) -> str:
    """Build the prompt without few-shot examples (zero-shot)."""
    parts = [SYSTEM_PROMPT_ZEROSHOT]
    parts.append(f"Problem: {question}")
    return "\n\n".join(parts)


def make_prefix_fewshot(dp, fewshot_examples: str):
    """Create prefix with few-shot examples."""
    question = dp['question'].strip()
    if question[-1] != '?':
        question += '?'
    
    return build_fewshot_prompt(question, fewshot_examples)


def make_prefix_zeroshot(dp):
    """Create prefix without few-shot examples (zero-shot)."""
    question = dp['question'].strip()
    if question[-1] != '?':
        question += '?'
    
    return build_zeroshot_prompt(question)


def make_prefix_base(dp):
    """Original base prefix without few-shot (for comparison)."""
    question = dp['question'].strip()
    if question[-1] != '?':
        question += '?'

    prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_search_fewshot')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='fewshot', 
                        choices=['fewshot', 'zeroshot', 'base'],
                        help='Template type: fewshot (with examples), zeroshot (no examples), or base (original Search-R1 format)')
    parser.add_argument('--examples_file', type=str, default=None,
                        help='Path to few-shot examples file')
    parser.add_argument('--num_examples', type=int, default=None,
                        help='Number of few-shot examples to include (None for all, 1, 2, 3, etc.)')
    parser.add_argument('--train_data_num', type=int, default=None,
                        help='Number of training samples to use (None for all)')
    parser.add_argument('--val_data_num', type=int, default=None,
                        help='Number of validation samples to use (None for all)')

    args = parser.parse_args()

    data_source = 'nq'

    # Load few-shot examples if using fewshot template
    fewshot_examples = ""
    if args.template_type == 'fewshot':
        fewshot_examples = load_fewshot_examples(args.examples_file, args.num_examples)
        num_ex_str = f"{args.num_examples} example(s)" if args.num_examples else "all examples"
        print(f"Loaded {num_ex_str} ({len(fewshot_examples)} chars)")
    elif args.template_type == 'zeroshot':
        print("Using zero-shot template (no examples)")
    else:
        print("Using base template (original Search-R1 format)")

    dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # Optionally limit dataset size
    if args.train_data_num is not None and args.train_data_num > 0:
        train_dataset = train_dataset.select(range(min(args.train_data_num, len(train_dataset))))
        print(f"Limited training data to {len(train_dataset)} samples")
    
    if args.val_data_num is not None and args.val_data_num > 0:
        test_dataset = test_dataset.select(range(min(args.val_data_num, len(test_dataset))))
        print(f"Limited validation data to {len(test_dataset)} samples")

    def make_map_fn(split, template_type, fewshot_examples, num_examples):
        def process_fn(example, idx):
            if template_type == 'fewshot':
                question = make_prefix_fewshot(example, fewshot_examples)
            elif template_type == 'zeroshot':
                question = make_prefix_zeroshot(example)
            else:
                question = make_prefix_base(example)
            
            solution = {
                "target": example['golden_answers'],
            }

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'template_type': template_type,
                    'num_examples': num_examples if template_type == 'fewshot' else 0,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(
        function=make_map_fn('train', args.template_type, fewshot_examples, args.num_examples), 
        with_indices=True
    )
    test_dataset = test_dataset.map(
        function=make_map_fn('test', args.template_type, fewshot_examples, args.num_examples), 
        with_indices=True
    )

    # Create output directory
    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    print(f"Saved {len(train_dataset)} train samples to {os.path.join(local_dir, 'train.parquet')}")
    print(f"Saved {len(test_dataset)} test samples to {os.path.join(local_dir, 'test.parquet')}")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(os.path.join(local_dir, 'train.parquet'), args.hdfs_dir)
        copy(os.path.join(local_dir, 'test.parquet'), args.hdfs_dir)
        print(f"Copied to HDFS: {args.hdfs_dir}")
