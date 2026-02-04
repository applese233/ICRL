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
Reward scoring functions for few-shot web search training.
Combines exact match accuracy with format compliance scoring.
"""

import re
import string
import random


def normalize_answer(s):
    """Normalize answer string for comparison."""
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


def em_check(prediction, golden_answers):
    """Check if prediction exactly matches any of the golden answers."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return True
    return False


def set_match_check(prediction, golden_answers):
    """Check if prediction contains ALL golden answers (order-insensitive).
    
    Useful for questions with multiple correct answers where order doesn't matter.
    E.g., "What are the five boroughs of NYC?" -> any order of the 5 boroughs is correct.
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    
    normalized_prediction = normalize_answer(prediction)
    
    # Check if all golden answers are contained in the prediction
    all_found = True
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) not in normalized_prediction:
            all_found = False
            break
    
    return all_found


def subem_check(prediction, golden_answers):
    """Check if any golden answer is a substring of the prediction."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) in normalized_prediction:
            return True
    return False


def extract_answer(solution_str):
    """Extract answer from <answer>...</answer> tags.
    
    Returns the content of the last <answer> tag if multiple exist.
    For few-shot training, we want at least 2 <answer> tags (from examples + actual).
    """
    if not solution_str:
        return None
    
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL | re.IGNORECASE))
    
    # For strict format checking with few-shot examples, we expect multiple answer tags
    # The last one should be the model's actual answer
    if len(matches) <= 1:
        # If only 0-1 matches, the model didn't generate a proper answer
        return None
    
    # Return the last match (the model's actual answer)
    return matches[-1].group(1).strip()


def extract_answer_flexible(solution_str):
    """Extract answer more flexibly - returns last answer tag content or last line."""
    if not solution_str:
        return ""
    
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL | re.IGNORECASE))
    
    if matches:
        return matches[-1].group(1).strip()
    
    # Fallback: return last non-empty line
    lines = [ln.strip() for ln in solution_str.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def compute_format_score(solution_str, return_stats=False):
    """Compute format compliance score.
    
    Rewards proper use of:
    - <think>...</think> tags for reasoning
    - <search>...</search> tags for queries
    - <answer>...</answer> tags for final answer
    
    Returns a score between 0.0 and 1.0
    If return_stats=True, also returns a dict with tag counts
    """
    if not solution_str:
        if return_stats:
            return 0.0, {'search_count': 0, 'think_count': 0, 'answer_count': 0}
        return 0.0
    
    text_lower = solution_str.lower()
    score = 1.0
    
    # Check for answer tags (most important)
    answer_open_cnt = text_lower.count("<answer>")
    answer_close_cnt = text_lower.count("</answer>")
    
    # Must have at least one answer tag pair
    if answer_open_cnt == 0 or answer_close_cnt == 0:
        score -= 0.5
    
    # Answer tags should be balanced
    if answer_open_cnt != answer_close_cnt:
        score -= 0.2
    
    # Check for think tags (reasoning)
    think_open_cnt = text_lower.count("<think>")
    think_close_cnt = text_lower.count("</think>")
    
    if think_open_cnt == 0 or think_close_cnt == 0:
        score -= 0.15
    
    if think_open_cnt != think_close_cnt:
        score -= 0.1
    
    # Check for search usage (soft requirement - we want the model to learn to search)
    search_open_cnt = text_lower.count("<search>")
    search_close_cnt = text_lower.count("</search>")
    has_search = search_open_cnt > 0 or search_close_cnt > 0
    
    if not has_search:
        score -= 0.1  # Small penalty for not using search
    
    # Check for extracted answer content
    extracted = extract_answer_flexible(solution_str)
    if not extracted:
        score -= 0.2
    
    final_score = max(0.0, min(1.0, score))
    
    if return_stats:
        stats = {
            'search_count': min(search_open_cnt, search_close_cnt),  # Count complete search pairs
            'think_count': min(think_open_cnt, think_close_cnt),
            'answer_count': min(answer_open_cnt, answer_close_cnt),
        }
        return final_score, stats
    
    return final_score


def compute_accuracy(solution_str, ground_truth):
    """Compute accuracy with strict matching.
    
    Tries in order:
    1. Exact match (EM) - prediction matches one of the answers exactly
    2. Set match - for multi-answer questions, ALL answers are contained in prediction
    
    Args:
        solution_str: The full solution string
        ground_truth: Dict with 'target' key containing list of acceptable answers
        
    Returns:
        1.0 if exact/set match, 0.0 otherwise
    """
    answer = extract_answer(solution_str)
    
    if answer is None:
        # Try flexible extraction as fallback
        answer = extract_answer_flexible(solution_str)
        if not answer:
            return 0.0
    
    targets = ground_truth.get('target', [])
    if isinstance(targets, str):
        targets = [targets]
    
    # Strategy 1: Exact match with any single target
    if em_check(answer, targets):
        return 1.0
    
    # Strategy 2: Set match - all targets are contained in the answer
    # This handles cases like "The Bronx, Brooklyn, Manhattan, Queens, and Staten Island"
    # when targets are ['Manhattan', 'Queens', 'Staten Island', 'Brooklyn', 'the Bronx']
    if len(targets) > 1 and set_match_check(answer, targets):
        return 1.0
    
    return 0.0


def compute_score_fewshot(solution_str, ground_truth, 
                          accuracy_weight=0.6, 
                          format_weight=0.4,
                          format_score=0.0,
                          return_details=False):
    """Compute combined reward score for few-shot web search training.
    
    Args:
        solution_str: The full solution string
        ground_truth: Dict with 'target' key containing list of acceptable answers
        accuracy_weight: Weight for exact match accuracy (default 0.6)
        format_weight: Weight for format compliance (default 0.4)
        format_score: Base score for correct format but wrong answer (default 0.0)
        return_details: If True, return (score, accuracy, format, extracted_answer, stats)
        
    Returns:
        If return_details=False: Combined score between 0.0 and 1.0
        If return_details=True: (score, accuracy, format_score, extracted_answer, stats)
    """
    do_print = random.randint(1, 64) <= 2
    
    acc = compute_accuracy(solution_str, ground_truth)
    fmt, stats = compute_format_score(solution_str, return_stats=True)
    extracted_answer = extract_answer(solution_str) or extract_answer_flexible(solution_str)
    
    # Combine scores
    score = accuracy_weight * acc + format_weight * fmt
    
    # Ensure score is in valid range
    final_score = max(0.0, min(1.0, score))
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth.get('target', [])}")
        print(f"Extracted answer: {extracted_answer}")
        print(f"Accuracy: {acc}, Format: {fmt}, Final: {final_score}")
        print(f"Search calls: {stats['search_count']}, Think calls: {stats['think_count']}")
        # Only print first 500 chars of solution to avoid clutter
        print(f"Solution (truncated): {solution_str[:500]}...")
    
    if return_details:
        return final_score, acc, fmt, extracted_answer, stats
    return final_score


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """Original EM scoring function for backward compatibility.
    
    Uses the same logic as the original qa_em.py for comparison.
    """
    answer = extract_answer(solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str[:500]}...")
    
    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth['target']):
            return score
        else:
            return format_score


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """Substring EM scoring function."""
    answer = extract_answer(solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score
