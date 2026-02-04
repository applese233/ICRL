"""
Single Question Inference Script for ICRL

This script allows you to ask a single question and see how the model
reasons and searches to find the answer.

Usage:
    # Interactive mode (asks for question input)
    python infer.py --model_path Qwen/Qwen2.5-7B-Instruct
    
    # With a specific question
    python infer.py --model_path Qwen/Qwen2.5-7B-Instruct --question "What is the capital of France?"
    
    # Use few-shot prompts
    python infer.py --model_path your_model --use_fewshot --fewshot_path example/fewshot_examples.txt
"""

import argparse
import re
import torch
import transformers
import requests
from typing import Optional


def parse_args():
    parser = argparse.ArgumentParser(description="Single question inference with search-augmented LLM")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model (local path or HuggingFace model ID)")
    parser.add_argument("--question", type=str, default=None,
                        help="Question to ask. If not provided, will prompt for input.")
    parser.add_argument("--search_url", type=str, default="http://127.0.0.1:8000/retrieve",
                        help="URL of the search server")
    parser.add_argument("--topk", type=int, default=3,
                        help="Number of search results to retrieve")
    parser.add_argument("--max_turns", type=int, default=10,
                        help="Maximum number of search turns")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Maximum new tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--use_fewshot", action="store_true",
                        help="Use few-shot examples in the prompt")
    parser.add_argument("--fewshot_path", type=str, default="example/fewshot_examples.txt",
                        help="Path to few-shot examples file")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed generation process")
    return parser.parse_args()


# Prompt templates
SYSTEM_PROMPT = "You are a helpful assistant."

ZERO_SHOT_PROMPT = """Solve the following problem step by step. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.

Question: {question}"""

FEWSHOT_PROMPT_TEMPLATE = """Solve the following problem step by step. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations.

Here are some examples:

{fewshot_examples}

Now solve the following question:

Question: {question}"""


def load_fewshot_examples(fewshot_path: str) -> str:
    """Load few-shot examples from file."""
    try:
        with open(fewshot_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"[WARNING] Few-shot examples file not found: {fewshot_path}")
        return ""


def get_query(text: str) -> Optional[str]:
    """Extract search query from model output."""
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    return None


def get_answer(text: str) -> Optional[str]:
    """Extract final answer from model output."""
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    return None


def search(query: str, search_url: str, topk: int = 3) -> str:
    """Call the search server and format results."""
    try:
        payload = {
            "queries": [query],
            "topk": topk,
            "return_scores": True
        }
        response = requests.post(search_url, json=payload, timeout=30)
        results = response.json()['result']
        
        def _passages2string(retrieval_result):
            format_reference = ''
            for idx, doc_item in enumerate(retrieval_result):
                content = doc_item['document']['contents']
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
            return format_reference
        
        return _passages2string(results[0])
    except Exception as e:
        print(f"[ERROR] Search failed: {e}")
        return "Search failed. Please try a different query."


class StopOnSequence(transformers.StoppingCriteria):
    """Custom stopping criteria to stop on specific sequences."""
    
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [tokenizer.encode(seq, add_special_tokens=False) for seq in target_sequences]
        self.target_lengths = [len(ids) for ids in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        targets = [torch.as_tensor(ids, device=input_ids.device) for ids in self.target_ids]
        
        if input_ids.shape[1] < min(self.target_lengths):
            return False
        
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True
        return False


def run_inference(
    model,
    tokenizer,
    question: str,
    search_url: str,
    topk: int = 3,
    max_turns: int = 10,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    use_fewshot: bool = False,
    fewshot_examples: str = "",
    verbose: bool = True
):
    """Run inference for a single question."""
    
    device = next(model.parameters()).device
    
    # Build prompt
    if use_fewshot and fewshot_examples:
        user_prompt = FEWSHOT_PROMPT_TEMPLATE.format(
            fewshot_examples=fewshot_examples,
            question=question
        )
    else:
        user_prompt = ZERO_SHOT_PROMPT.format(question=question)
    
    # Apply chat template
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_prompt}\n\nAssistant:"
    
    # Stopping criteria
    target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n"]
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])
    
    # EOS tokens for Qwen2.5 series
    eos_token_ids = [151645, 151643]
    if tokenizer.eos_token_id:
        eos_token_ids.append(tokenizer.eos_token_id)
    
    search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'
    
    full_response = ""
    search_count = 0
    
    if verbose:
        print("\n" + "="*60)
        print("Question:", question)
        print("="*60 + "\n")
    
    for turn in range(max_turns):
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True if temperature > 0 else False,
                temperature=temperature if temperature > 0 else None,
            )
        
        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        if verbose:
            print(output_text, end="")
        
        full_response += output_text
        
        # Check if generation is complete
        if outputs[0][-1].item() in eos_token_ids:
            break
        
        # Check for search query
        query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if query:
            search_count += 1
            if verbose:
                print(f"\n[Searching: {query}]")
            
            search_results = search(query, search_url, topk)
            search_text = search_template.format(output_text=output_text, search_results=search_results)
            prompt += search_text
            
            if verbose:
                print(f"<information>{search_results}</information>\n")
            
            full_response += f"<information>{search_results}</information>\n\n"
        else:
            break
    
    # Extract final answer
    answer = get_answer(full_response)
    
    if verbose:
        print("\n" + "="*60)
        print(f"Final Answer: {answer}")
        print(f"Total searches: {search_count}")
        print("="*60)
    
    return {
        "question": question,
        "answer": answer,
        "full_response": full_response,
        "search_count": search_count
    }


def main():
    args = parse_args()
    
    print(f"Loading model: {args.model_path}")
    
    # Load tokenizer and model
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    # Load few-shot examples if needed
    fewshot_examples = ""
    if args.use_fewshot:
        fewshot_examples = load_fewshot_examples(args.fewshot_path)
        if fewshot_examples:
            print(f"Loaded few-shot examples from: {args.fewshot_path}")
    
    # Get question
    if args.question:
        question = args.question
    else:
        print("\nEnter your question (or 'quit' to exit):")
        question = input("> ").strip()
    
    while question.lower() not in ['quit', 'exit', 'q']:
        if question:
            result = run_inference(
                model=model,
                tokenizer=tokenizer,
                question=question,
                search_url=args.search_url,
                topk=args.topk,
                max_turns=args.max_turns,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                use_fewshot=args.use_fewshot,
                fewshot_examples=fewshot_examples,
                verbose=args.verbose or True
            )
        
        if args.question:
            break
        
        print("\nEnter your question (or 'quit' to exit):")
        question = input("> ").strip()
    
    print("Goodbye!")


if __name__ == "__main__":
    main()
