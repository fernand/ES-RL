import argparse
import json
import logging
import shutil
import os
from typing import List, Dict, Tuple

import datasets
import numpy as np
import torch
from unsloth import FastLanguageModel
from tqdm import tqdm

from reward import format_reward_func, equation_reward_func
from lm_cma_es import LMCMAES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LoRAWeightManager:
    """Manages LoRA weights for CMA-ES optimization"""

    def __init__(self, model_name: str, rank: int = 16, alpha: int = 32, dtype=torch.bfloat16):
        self.model_name = model_name
        self.rank = rank
        self.alpha = alpha
        self.dtype = dtype

        # Define LoRA target modules for Qwen2.5 attention layers only
        # Unsloth uses different naming conventions
        self.target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ]

        # Get model config to determine layer dimensions
        # Qwen2.5-3B has 36 layers, hidden_size=2048, num_heads=16, num_kv_heads=2 (GQA)
        self.num_layers = 36
        self.hidden_size = 2048
        self.num_heads = 16
        self.num_kv_heads = 2  # Grouped Query Attention
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim  # 2 * 128 = 256

        # Calculate total parameter count for LoRA
        self.param_shapes = self._get_param_shapes()
        self.total_params = sum(np.prod(shape) for shape in self.param_shapes.values())
        logger.info(f"Total LoRA parameters: {self.total_params:,}")

    def _get_param_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Get shapes for all LoRA parameters"""
        shapes = {}

        for layer_idx in range(self.num_layers):
            for module in self.target_modules:
                # Determine input/output dimensions based on module type
                in_features = self.hidden_size  # All modules have same input dim

                if module == "q_proj":
                    out_features = self.hidden_size  # 2048
                elif module in ["k_proj", "v_proj"]:
                    out_features = self.kv_dim  # 256 (due to GQA)
                elif module == "o_proj":
                    out_features = self.hidden_size  # 2048

                # Standard PEFT naming convention (without the .default suffix)
                key_a = f"base_model.model.model.layers.{layer_idx}.self_attn.{module}.lora_A.weight"
                key_b = f"base_model.model.model.layers.{layer_idx}.self_attn.{module}.lora_B.weight"
                # Standard LoRA shapes (not transposed like vLLM)
                shapes[key_a] = (self.rank, in_features)  # (16, 2048)
                shapes[key_b] = (out_features, self.rank)  # (2048, 16)

        return shapes

    def init_weights(self) -> np.ndarray:
        """Initialize LoRA weights using Kaiming initialization"""
        weights = []

        for key, shape in self.param_shapes.items():
            if "lora_A" in key:
                # Initialize A matrix with Kaiming uniform
                # shape is now (rank, in_features), so scale by input dimension shape[1]
                w = np.random.randn(*shape) * np.sqrt(2.0 / shape[1])
            else:  # lora_B
                # Initialize B matrix to zeros
                w = np.zeros(shape)
            weights.append(w.flatten())

        return np.concatenate(weights).astype(np.float32)

    def vector_to_lora_dict(self, vector: np.ndarray) -> Dict[str, torch.Tensor]:
        """Convert flat vector to LoRA weight dictionary"""
        weights_dict = {}
        offset = 0

        for key, shape in self.param_shapes.items():
            size = np.prod(shape)
            w = vector[offset:offset + size].reshape(shape)
            weights_dict[key] = torch.tensor(w, dtype=self.dtype)
            offset += size

        return weights_dict

    def save_lora_adapter(self, weights_dict: Dict[str, torch.Tensor], save_path: str):
        """Save LoRA adapter in format compatible with PEFT/Unsloth"""
        os.makedirs(save_path, exist_ok=True)

        # Save adapter config in PEFT format
        adapter_config = {
            "r": self.rank,
            "lora_alpha": self.alpha,
            "target_modules": self.target_modules,
            "lora_dropout": 0.0,
            "base_model_name_or_path": self.model_name,
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "bias": "none",
        }

        with open(os.path.join(save_path, "adapter_config.json"), "w") as f:
            json.dump(adapter_config, f, indent=2)

        # Save weights in safetensors format
        from safetensors.torch import save_file
        save_file(weights_dict, os.path.join(save_path, "adapter_model.safetensors"))


class LMMAESTrainer:
    """LM-MA-ES trainer for LoRA optimization with Unsloth"""

    def __init__(
        self,
        model_name: str,
        dataset_path: str,
        population_size: int = 92,
        rank: int = 16,
        alpha: int = 32,
        batch_size: int = 32,
        max_generations: int = 100,
        eval_samples: int = 256,
        sigma: float = 0.1,
        seed: int = 42,
        checkpoint_dir: str = "checkpoints",
        temp_dir: str = "temp_lora",
    ):
        self.model_name = model_name
        self.population_size = population_size
        self.batch_size = batch_size
        self.max_generations = max_generations
        self.eval_samples = eval_samples
        self.sigma = sigma
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir
        self.temp_dir = temp_dir

        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize LoRA weight manager
        self.lora_manager = LoRAWeightManager(model_name, rank=rank, alpha=alpha)

        # Load dataset
        logger.info(f"Loading dataset from {dataset_path}")
        self.dataset = datasets.load_from_disk(dataset_path)
        self.train_data = self.dataset["train"]
        self.eval_data = self.dataset["test"]

        # Initialize Unsloth model
        logger.info("Initializing Unsloth model")
        self.base_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )

        # Set up tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Generation parameters
        self.generation_kwargs = {
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,  # Will add stop tokens in generation
        }

        # Create PEFT model once for reuse
        logger.info("Creating PEFT model for LoRA adaptation")
        self.peft_model = FastLanguageModel.get_peft_model(
            self.base_model,
            r=rank,
            target_modules=self.lora_manager.target_modules,
            lora_alpha=alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=False,
        )
        FastLanguageModel.for_inference(self.peft_model)

        # Initialize LM-CMA-ES
        initial_weights = self.lora_manager.init_weights()
        # Use our custom LM-CMA-ES for better performance on high-dimensional problems
        self.es = LMCMAES(
            initial_weights,
            self.sigma,
            inopts={
                'popsize': self.population_size,
                'seed': self.seed,
                'bounds': [-5, 5],  # Bounded optimization for stability
                'memory_limit': int(4 + 3 * np.log(len(initial_weights))),  # Limited memory budget
            }
        )

        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        # Track best solution
        self.best_reward = -float('inf')
        self.best_weights = None

    def evaluate_population(self, population: List[np.ndarray]) -> List[float]:
        """Evaluate a population of LoRA weights"""
        all_rewards = []

        # Sample evaluation data
        eval_indices = np.random.choice(len(self.train_data), self.eval_samples, replace=False)
        eval_batch = self.train_data.select(eval_indices)

        # Process each member of the population
        for idx, weights in enumerate(tqdm(population, desc="Evaluating population")):
            # Convert weights to LoRA dict
            lora_dict = self.lora_manager.vector_to_lora_dict(weights)

            # Update the LoRA weights directly in the model
            with torch.no_grad():
                for name, param in self.peft_model.named_parameters():
                    if "lora_" in name:
                        # Build the expected key based on the parameter name
                        # Remove 'default.' if present and ensure correct format
                        clean_name = name.replace(".default", "")
                        if clean_name in lora_dict:
                            param.data.copy_(lora_dict[clean_name])
                        elif name in lora_dict:
                            param.data.copy_(lora_dict[name])

            model = self.peft_model

            # Generate completions in batches
            prompts = eval_batch["prompt"]
            targets = eval_batch["target"]
            nums = eval_batch["nums"]

            completions = []
            for i in range(0, len(prompts), self.batch_size):
                batch_prompts = prompts[i:i + self.batch_size]

                # Tokenize inputs
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to(model.device)

                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        **self.generation_kwargs
                    )

                # Decode outputs (skip input tokens)
                batch_completions = []
                for j, output in enumerate(outputs):
                    input_length = inputs.input_ids[j].shape[0]
                    generated_tokens = output[input_length:]
                    completion = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    batch_completions.append(completion)

                completions.extend(batch_completions)

            # Compute rewards
            format_rewards = format_reward_func(completions, targets)
            equation_rewards = equation_reward_func(completions, targets, nums)

            # Combine rewards (average of format and equation correctness)
            combined_rewards = [
                0.3 * f + 0.7 * e
                for f, e in zip(format_rewards, equation_rewards)
            ]

            # Average reward for this population member
            avg_reward = np.mean(combined_rewards)
            all_rewards.append(avg_reward)

            # Clear GPU memory periodically
            if idx % 10 == 0:
                torch.cuda.empty_cache()

        return all_rewards

    def train(self):
        """Main training loop"""
        logger.info("Starting LM-MA-ES training")

        for generation in range(self.max_generations):
            if generation >= self.max_generations or self.es.stop():
                logger.info(f"Stopping: generation={generation}, max={self.max_generations}")
                break

            # Sample population
            population = self.es.ask()

            # Evaluate population
            logger.info(f"Generation {generation + 1}/{self.max_generations}")
            rewards = self.evaluate_population(population)

            # Negate rewards for minimization (LM-MA-ES minimizes by default)
            costs = [-r for r in rewards]

            # Update LM-MA-ES
            self.es.tell(population, costs)

            # Track best solution
            best_idx = np.argmax(rewards)
            if rewards[best_idx] > self.best_reward:
                self.best_reward = rewards[best_idx]
                self.best_weights = population[best_idx]

                # Save best checkpoint
                self.save_checkpoint(generation, self.best_weights, self.best_reward)

            # Log statistics
            logger.info(f"Reward stats - Mean: {np.mean(rewards):.4f}, "
                       f"Max: {np.max(rewards):.4f}, Min: {np.min(rewards):.4f}, "
                       f"Std: {np.std(rewards):.4f}")
            logger.info(f"Best reward so far: {self.best_reward:.4f}")

            # Periodic evaluation on test set
            if (generation + 1) % 10 == 0:
                self.evaluate_on_test_set(self.best_weights)

    def evaluate_on_test_set(self, weights: np.ndarray):
        """Evaluate best weights on test set"""
        logger.info("Evaluating on test set...")

        # Convert weights to LoRA dict
        lora_dict = self.lora_manager.vector_to_lora_dict(weights)

        # Update the LoRA weights directly in the reusable PEFT model
        with torch.no_grad():
            for name, param in self.peft_model.named_parameters():
                if "lora_" in name:
                    clean_name = name.replace(".default", "")
                    if clean_name in lora_dict:
                        param.data.copy_(lora_dict[clean_name])
                    elif name in lora_dict:
                        param.data.copy_(lora_dict[name])

        model = self.peft_model

        # Sample from test set
        test_samples = min(500, len(self.eval_data))
        test_indices = np.random.choice(len(self.eval_data), test_samples, replace=False)
        test_batch = self.eval_data.select(test_indices)

        # Generate completions
        prompts = test_batch["prompt"]
        targets = test_batch["target"]
        nums = test_batch["nums"]

        completions = []
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]

            # Tokenize inputs
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(model.device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **self.generation_kwargs
                )

            # Decode outputs (skip input tokens)
            batch_completions = []
            for j, output in enumerate(outputs):
                input_length = inputs.input_ids[j].shape[0]
                generated_tokens = output[input_length:]
                completion = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                batch_completions.append(completion)

            completions.extend(batch_completions)

        # Compute rewards
        format_rewards = format_reward_func(completions, targets)
        equation_rewards = equation_reward_func(completions, targets, nums)

        logger.info(f"Test set results - Format accuracy: {np.mean(format_rewards):.4f}, "
                   f"Equation accuracy: {np.mean(equation_rewards):.4f}")

        torch.cuda.empty_cache()

    def save_checkpoint(self, generation: int, weights: np.ndarray, reward: float):
        """Save checkpoint with LoRA weights"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"generation_{generation + 1}")

        # Convert weights to LoRA dict and save
        lora_dict = self.lora_manager.vector_to_lora_dict(weights)
        self.lora_manager.save_lora_adapter(lora_dict, checkpoint_path)

        # Save metadata
        metadata = {
            "generation": generation + 1,
            "reward": float(reward),
            "population_size": self.population_size,
            "rank": self.lora_manager.rank,
            "alpha": self.lora_manager.alpha,
            "model_name": self.model_name,
        }

        with open(os.path.join(checkpoint_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Also save as "best" checkpoint
        best_path = os.path.join(self.checkpoint_dir, "best")
        if os.path.exists(best_path):
            shutil.rmtree(best_path)
        shutil.copytree(checkpoint_path, best_path)


def main():
    parser = argparse.ArgumentParser(description="LM-MA-ES training for LoRA optimization with Unsloth")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
                       help="Model name or path")
    parser.add_argument("--dataset_path", type=str, default="countdown_dataset",
                       help="Path to processed dataset")
    parser.add_argument("--population_size", type=int, default=92,
                       help="CMA-ES population size")
    parser.add_argument("--rank", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=32,
                       help="LoRA alpha scaling factor")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size for inference")
    parser.add_argument("--max_generations", type=int, default=100,
                       help="Maximum number of CMA-ES generations")
    parser.add_argument("--eval_samples", type=int, default=256,
                       help="Number of samples to evaluate per population member")
    parser.add_argument("--sigma", type=float, default=0.1,
                       help="Initial sigma for CMA-ES")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints")

    args = parser.parse_args()

    # Create trainer and start training
    trainer = LMMAESTrainer(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        population_size=args.population_size,
        rank=args.rank,
        alpha=args.alpha,
        batch_size=args.batch_size,
        max_generations=args.max_generations,
        eval_samples=args.eval_samples,
        sigma=args.sigma,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
    )

    trainer.train()


if __name__ == "__main__":
    main()