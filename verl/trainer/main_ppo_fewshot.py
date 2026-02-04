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
Main PPO trainer for few-shot web search learning.
Based on main_ppo.py but with support for format+accuracy combined reward.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.utils.reward_score import qa_em_fewshot
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np


def _select_rm_score_fn(data_source, reward_type='em'):
    """Select reward scoring function based on data source and reward type.
    
    Args:
        data_source: The data source name
        reward_type: 'fewshot' for format+accuracy reward, 'em' for exact match only
        
    Returns:
        Scoring function
    """
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        if reward_type == 'fewshot':
            return qa_em_fewshot.compute_score_fewshot
        else:
            return qa_em.compute_score_em
    else:
        # Default to fewshot for unknown data sources
        if reward_type == 'fewshot':
            return qa_em_fewshot.compute_score_fewshot
        return qa_em.compute_score_em


class RewardManager():
    """The reward manager with support for few-shot format+accuracy reward."""

    def __init__(self, tokenizer, num_examine, format_score=0., 
                 reward_type='em', accuracy_weight=0.6, format_weight=0.4) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score
        self.reward_type = reward_type
        self.accuracy_weight = accuracy_weight
        self.format_weight = format_weight

    def __call__(self, data: DataProto, collect_samples=False, max_samples=10):
        """Compute rewards for the batch.
        
        Args:
            data: DataProto containing batch data
            collect_samples: Whether to collect detailed sample info for logging
            max_samples: Maximum number of samples to collect for logging
            
        Returns:
            If collect_samples=False: reward_tensor
            If collect_samples=True: (reward_tensor, sample_list, metrics_dict)
        """

        # If there is rm score, we directly return rm score
        if 'rm_scores' in data.batch.keys():
            if collect_samples:
                return data.batch['rm_scores'], [], {}
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        
        # For collecting samples
        collected_samples = []
        total_accuracy = 0.0
        total_format = 0.0
        total_score = 0.0
        total_search_count = 0
        total_think_count = 0
        sample_count = 0

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode sequences
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            
            # Decode prompt and response separately for logging
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']

            # Select and compute reward
            compute_score_fn = _select_rm_score_fn(data_source, self.reward_type)

            if self.reward_type == 'fewshot':
                score, acc, fmt, extracted_answer, stats = compute_score_fn(
                    solution_str=sequences_str, 
                    ground_truth=ground_truth,
                    accuracy_weight=self.accuracy_weight,
                    format_weight=self.format_weight,
                    format_score=self.format_score,
                    return_details=True
                )
            else:
                score = compute_score_fn(
                    solution_str=sequences_str, 
                    ground_truth=ground_truth, 
                    format_score=self.format_score
                )
                acc = 1.0 if score > 0 else 0.0
                fmt = 1.0
                extracted_answer = ""
                stats = {'search_count': 0, 'think_count': 0, 'answer_count': 0}

            reward_tensor[i, valid_response_length - 1] = score
            
            # Accumulate metrics
            total_accuracy += acc
            total_format += fmt
            total_score += score
            total_search_count += stats.get('search_count', 0)
            total_think_count += stats.get('think_count', 0)
            sample_count += 1

            # Print samples for debugging
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"[RewardManager] Sample from {data_source}:")
                print(f"Score: {score}")
                print(sequences_str[:1000] + "..." if len(sequences_str) > 1000 else sequences_str)
            
            # Collect samples for wandb logging (random sampling)
            if collect_samples and len(collected_samples) < max_samples:
                # Sample randomly - collect with probability max_samples/batch_size
                import random
                if random.random() < max_samples / len(data):
                    sample_info = {
                        'data_source': data_source,
                        'ground_truth': str(ground_truth.get('target', [])),
                        'extracted_answer': str(extracted_answer) if extracted_answer else "",
                        'accuracy': acc,
                        'format_score': fmt,
                        'total_score': score,
                        'prompt': prompt_str[-2000:] if len(prompt_str) > 2000 else prompt_str,  # Truncate long prompts
                        'response': response_str[-3000:] if len(response_str) > 3000 else response_str,  # Truncate long responses
                    }
                    collected_samples.append(sample_info)

        if collect_samples:
            metrics_dict = {
                'avg_accuracy': total_accuracy / max(sample_count, 1),
                'avg_format_score': total_format / max(sample_count, 1),
                'avg_total_score': total_score / max(sample_count, 1),
                'avg_search_count': total_search_count / max(sample_count, 1),
                'avg_think_count': total_think_count / max(sample_count, 1),
                'total_search_count': total_search_count,
                'total_think_count': total_think_count,
            }
            return reward_tensor, collected_samples, metrics_dict
            
        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # Print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # Download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # Instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # Define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # Handle reward model if enabled
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # Get reward configuration (with defaults for backward compatibility)
    reward_type = config.get('reward', {}).get('type', 'em')
    accuracy_weight = config.get('reward', {}).get('accuracy_weight', 0.6)
    format_weight = config.get('reward', {}).get('format_weight', 0.4)
    
    print(f"[main_ppo_fewshot] Reward config: type={reward_type}, "
          f"accuracy_weight={accuracy_weight}, format_weight={format_weight}")

    # Create reward managers with fewshot support
    reward_fn = RewardManager(
        tokenizer=tokenizer, 
        num_examine=0,
        reward_type=reward_type,
        accuracy_weight=accuracy_weight,
        format_weight=format_weight
    )

    val_reward_fn = RewardManager(
        tokenizer=tokenizer, 
        num_examine=1,
        reward_type=reward_type,
        accuracy_weight=accuracy_weight,
        format_weight=format_weight
    )

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
