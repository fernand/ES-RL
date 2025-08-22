import argparse
import json
import shutil
import os
from typing import List, Dict, Tuple, Optional
import math

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from unsloth import FastLanguageModel
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from reward import format_reward_func, equation_reward_func


class GRPOTrainer:
    """GRPO (Group Relative Policy Optimization) trainer for LoRA fine-tuning"""

    def __init__(
        self,
        model_name: str,
        dataset_path: str,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        rank: int = 16,
        alpha: int = 32,
        max_epochs: int = 3,
        eval_samples: int = 256,
        seed: int = 42,
        checkpoint_dir: str = "checkpoints_grpo",
        kl_coef: float = 0.05,
        clip_range: float = 0.2,
        group_size: int = 4,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.eval_samples = eval_samples
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir
        self.kl_coef = kl_coef
        self.clip_range = clip_range
        self.group_size = group_size  # Number of samples per prompt for GRPO

        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Load dataset
        print(f"Loading dataset from {dataset_path}")
        self.dataset = datasets.load_from_disk(dataset_path)
        self.train_data = self.dataset["train"]
        self.eval_data = self.dataset["test"]

        # Initialize model with Unsloth
        print("Initializing Unsloth model with LoRA")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )

        # Add LoRA adapters
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=seed,
        )

        # Set up tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Create reference model (frozen copy for KL divergence)
        print("Creating reference model for KL divergence")
        self.ref_model, _ = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        self.ref_model = FastLanguageModel.get_peft_model(
            self.ref_model,
            r=rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=False,
            random_state=seed,
        )
        FastLanguageModel.for_inference(self.ref_model)  # Freeze reference model

        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Track best model
        self.best_reward = -float('inf')

    def compute_rewards(self, completions: List[str], targets: List[int], nums: List[List[int]]) -> torch.Tensor:
        """Compute rewards for generated completions"""
        format_rewards = format_reward_func(completions, targets)
        equation_rewards = equation_reward_func(completions, targets, nums)

        # Combine rewards (30% format, 70% correctness)
        combined_rewards = [
            0.3 * f + 0.7 * e
            for f, e in zip(format_rewards, equation_rewards)
        ]

        return torch.tensor(combined_rewards, dtype=torch.float32)

    def generate_samples(self, prompts: List[str], num_samples: int = 1) -> Tuple[List[str], torch.Tensor]:
        """Generate multiple samples per prompt and return completions with log probs"""
        all_completions = []
        all_log_probs = []

        self.model.eval()
        with torch.no_grad():
            for prompt in prompts:
                # Repeat prompt for multiple samples
                batch_prompts = [prompt] * num_samples

                # Tokenize
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                )
                device = next(self.model.parameters()).device
                inputs = inputs.to(device)

                # Generate with sampling
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                # Extract completions
                for i in range(num_samples):
                    input_length = inputs.input_ids[i].shape[0]
                    generated_tokens = outputs.sequences[i][input_length:]
                    completion = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    all_completions.append(completion)

                    # Calculate log probabilities (simplified - using scores)
                    if outputs.scores:
                        log_probs = []
                        for j, score in enumerate(outputs.scores):
                            if j < len(generated_tokens) - 1:
                                token_id = generated_tokens[j + 1]
                                log_prob = F.log_softmax(score[i], dim=-1)[token_id]
                                log_probs.append(log_prob)
                        all_log_probs.append(torch.stack(log_probs).mean().item() if log_probs else 0.0)
                    else:
                        all_log_probs.append(0.0)

        return all_completions, torch.tensor(all_log_probs)

    def compute_grpo_loss(
        self,
        prompts: List[str],
        completions: List[str],
        rewards: torch.Tensor,
        old_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """Compute GRPO loss with group-relative rewards"""
        self.model.train()

        # Reshape rewards for group processing
        num_prompts = len(prompts)
        rewards = rewards.view(num_prompts, self.group_size)

        # Compute group-relative advantages (normalize within each group)
        advantages = rewards - rewards.mean(dim=1, keepdim=True)
        advantages = advantages / (advantages.std(dim=1, keepdim=True) + 1e-8)
        advantages = advantages.view(-1)  # Flatten back

        total_loss = 0
        num_batches = 0

        # Process all completions
        for i in range(0, len(completions), self.batch_size):
            batch_prompts = []
            batch_completions = completions[i:i + self.batch_size]
            batch_advantages = advantages[i:i + self.batch_size]

            # Expand prompts to match completions
            for j in range(len(batch_completions)):
                prompt_idx = (i + j) // self.group_size
                batch_prompts.append(prompts[prompt_idx])

            # Prepare inputs
            full_texts = [p + c for p, c in zip(batch_prompts, batch_completions)]

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )

            targets = self.tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1280  # 256 + 1024
            )

            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_advantages = batch_advantages.to(device)

            # Forward pass
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=targets.input_ids
            )

            # Compute policy gradient loss with advantages
            loss = outputs.loss

            # Weight by advantages (REINFORCE-style)
            weighted_loss = (loss * batch_advantages.detach()).mean()

            # Add KL penalty (simplified)
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    labels=targets.input_ids
                )

            kl_div = F.kl_div(
                F.log_softmax(outputs.logits, dim=-1),
                F.log_softmax(ref_outputs.logits, dim=-1),
                reduction='batchmean',
                log_target=True
            )

            # Final loss
            final_loss = weighted_loss + self.kl_coef * kl_div
            total_loss += final_loss.item()
            num_batches += 1

            # Backward pass with gradient accumulation
            # Scale loss by number of mini-batches for proper averaging
            num_minibatches = max(1, len(completions) // self.batch_size)
            (final_loss / num_minibatches).backward()

        return total_loss / max(num_batches, 1)

    def train_epoch(self, epoch: int):
        """Train for one epoch using GRPO"""
        print(f"\nEpoch {epoch + 1}/{self.max_epochs}")

        # Sample training data
        train_samples = min(self.eval_samples, len(self.train_data))
        indices = np.random.choice(len(self.train_data), train_samples, replace=False)
        batch = self.train_data.select(indices)

        prompts = batch["prompt"]
        targets = batch["target"]
        nums = batch["nums"]

        # Process in mini-batches
        total_loss = 0
        total_reward = 0
        num_batches = 0

        # Calculate prompts per batch to maintain total batch_size when expanded
        prompts_per_batch = max(1, self.batch_size // self.group_size)

        for i in tqdm(range(0, len(prompts), prompts_per_batch), desc="Training"):
            batch_prompts = prompts[i:i + prompts_per_batch]
            batch_targets = targets[i:i + prompts_per_batch]
            batch_nums = nums[i:i + prompts_per_batch]

            # Generate multiple samples per prompt
            completions, old_log_probs = self.generate_samples(batch_prompts, self.group_size)

            # Expand targets and nums to match completions
            expanded_targets = []
            expanded_nums = []
            for j in range(len(batch_targets)):
                expanded_targets.extend([batch_targets[j]] * self.group_size)
                expanded_nums.extend([batch_nums[j]] * self.group_size)

            # Compute rewards
            rewards = self.compute_rewards(completions, expanded_targets, expanded_nums)

            # Compute GRPO loss and update
            self.optimizer.zero_grad()
            loss = self.compute_grpo_loss(batch_prompts, completions, rewards, old_log_probs)
            self.optimizer.step()

            total_loss += loss
            total_reward += rewards.mean().item()
            num_batches += 1

            # Clear GPU memory periodically
            if i % 10 == 0:
                torch.cuda.empty_cache()

        avg_loss = total_loss / max(num_batches, 1)
        avg_reward = total_reward / max(num_batches, 1)

        print(f"Average loss: {avg_loss:.4f}")
        print(f"Average reward: {avg_reward:.4f}")

        return avg_reward

    def evaluate(self):
        """Evaluate model on test set"""
        print("Evaluating on test set...")

        # Sample test data
        test_samples = min(500, len(self.eval_data))
        indices = np.random.choice(len(self.eval_data), test_samples, replace=False)
        batch = self.eval_data.select(indices)

        prompts = batch["prompt"]
        targets = batch["target"]
        nums = batch["nums"]

        all_completions = []

        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(prompts), self.batch_size), desc="Evaluating"):
                batch_prompts = prompts[i:i + self.batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                )
                device = next(self.model.parameters()).device
                inputs = inputs.to(device)

                # Generate
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                # Extract completions
                for j, output in enumerate(outputs):
                    input_length = inputs.input_ids[j].shape[0]
                    generated_tokens = output[input_length:]
                    completion = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    all_completions.append(completion)

        # Compute rewards
        format_rewards = format_reward_func(all_completions, targets)
        equation_rewards = equation_reward_func(all_completions, targets, nums)

        print(f"Test set results - Format accuracy: {np.mean(format_rewards):.4f}, "
              f"Equation accuracy: {np.mean(equation_rewards):.4f}")

        combined_reward = np.mean([0.3 * f + 0.7 * e
                                   for f, e in zip(format_rewards, equation_rewards)])

        return combined_reward

    def save_checkpoint(self, epoch: int, reward: float):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch + 1}")

        print(f"Saving checkpoint to {checkpoint_path}")
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        # Save metadata
        metadata = {
            "epoch": epoch + 1,
            "reward": float(reward),
            "model_name": self.model_name,
        }

        with open(os.path.join(checkpoint_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Save as best if improved
        if reward > self.best_reward:
            self.best_reward = reward
            best_path = os.path.join(self.checkpoint_dir, "best")
            if os.path.exists(best_path):
                shutil.rmtree(best_path)
            shutil.copytree(checkpoint_path, best_path)
            print(f"New best model saved with reward: {reward:.4f}")

    def train(self):
        """Main training loop"""
        print("Starting GRPO training")
        print(f"Training for {self.max_epochs} epochs")
        print(f"Group size: {self.group_size}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")

        for epoch in range(self.max_epochs):
            # Train for one epoch
            avg_reward = self.train_epoch(epoch)

            # Evaluate on test set
            test_reward = self.evaluate()

            # Save checkpoint
            self.save_checkpoint(epoch, test_reward)

            print(f"Epoch {epoch + 1} completed. Test reward: {test_reward:.4f}")
            print(f"Best reward so far: {self.best_reward:.4f}")

        print(f"\nTraining completed. Best reward: {self.best_reward:.4f}")


def main():
    parser = argparse.ArgumentParser(description="GRPO training for LoRA fine-tuning with Unsloth")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
                       help="Model name or path")
    parser.add_argument("--dataset_path", type=str, default="countdown_dataset",
                       help="Path to processed dataset")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--rank", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=32,
                       help="LoRA alpha scaling factor")
    parser.add_argument("--max_epochs", type=int, default=3,
                       help="Maximum number of training epochs")
    parser.add_argument("--eval_samples", type=int, default=256,
                       help="Number of samples to use for training per epoch")
    parser.add_argument("--kl_coef", type=float, default=0.05,
                       help="KL divergence coefficient")
    parser.add_argument("--group_size", type=int, default=8,
                       help="Number of samples per prompt for GRPO")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_grpo",
                       help="Directory to save checkpoints")

    args = parser.parse_args()

    # Create trainer and start training
    trainer = GRPOTrainer(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        rank=args.rank,
        alpha=args.alpha,
        max_epochs=args.max_epochs,
        eval_samples=args.eval_samples,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        kl_coef=args.kl_coef,
        group_size=args.group_size,
    )

    trainer.train()


if __name__ == "__main__":
    main()