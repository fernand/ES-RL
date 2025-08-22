import argparse
import json
import shutil
import os
from typing import List, Tuple

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from unsloth import FastLanguageModel
from tqdm import tqdm

from reward import format_reward_func, equation_reward_func


class GRPOTrainer:
    """GRPO (Group Relative Policy Optimization) trainer for LoRA fine-tuning"""

    def __init__(
        self,
        model_name: str,
        dataset_path: str,
        batch_size: int = 32,
        learning_rate: float = 5e-5,
        rank: int = 16,
        alpha: int = 32,
        max_epochs: int = 3,
        seed: int = 42,
        checkpoint_dir: str = "checkpoints_grpo",
        kl_coef: float = 0.01,
        group_size: int = 8,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir
        self.kl_coef = kl_coef
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

        # Create reference model (frozen base model without LoRA for KL divergence)
        print("Creating reference model for KL divergence")
        self.ref_model, _ = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=torch.bfloat16,
            load_in_4bit=True,
            max_lora_rank=rank,
        )
        # Don't add LoRA to reference model - we want to compare against the base model
        FastLanguageModel.for_inference(self.ref_model)  # Freeze reference model

        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01
        )

        # Setup cosine annealing scheduler
        # Total steps = (train_data_size / (batch_size / group_size)) * max_epochs
        train_size = len(self.train_data)
        prompts_per_batch = max(1, batch_size // group_size)
        steps_per_epoch = max(1, train_size // prompts_per_batch)
        total_steps = steps_per_epoch * max_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=learning_rate * 0.1  # Min LR is 10% of initial LR
        )
        print(f"Cosine scheduler initialized: {steps_per_epoch} steps/epoch, {total_steps} total steps")
        print(f"Training on full dataset: {train_size} samples")

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

    def get_log_probs(self, model, prompts: List[str], completions: List[str], requires_grad: bool = False) -> torch.Tensor:
        """Calculate actual log probabilities for completions given prompts - batched version"""
        device = next(model.parameters()).device
        log_probs = []

        # Process in batches for efficiency using the configured batch_size
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i+self.batch_size]
            batch_completions = completions[i:i+self.batch_size]

            # Tokenize all prompts and full texts in batch
            prompt_tokens = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(device)

            full_texts = [p + c for p, c in zip(batch_prompts, batch_completions)]
            full_tokens = self.tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1280
            ).to(device)

            # Get model outputs for entire batch
            if requires_grad:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=full_tokens.input_ids,
                        attention_mask=full_tokens.attention_mask,
                    )
            else:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=full_tokens.input_ids,
                        attention_mask=full_tokens.attention_mask,
                    )

            # Process each item in batch
            for j in range(len(batch_prompts)):
                # Get actual lengths (handle padding)
                if self.tokenizer.padding_side == "left":
                    prompt_len = prompt_tokens.input_ids[j].shape[0]
                else:
                    prompt_len = int(prompt_tokens.attention_mask[j].sum().item())

                full_len = int(full_tokens.attention_mask[j].sum().item())

                # Extract log probs for generated tokens only
                generated_logits = outputs.logits[j, prompt_len-1:full_len-1]  # Shift by 1 for next token prediction
                generated_tokens = full_tokens.input_ids[j, prompt_len:full_len]

                # Calculate log probs
                token_log_probs = F.log_softmax(generated_logits, dim=-1)
                selected_log_probs = token_log_probs.gather(1, generated_tokens.unsqueeze(1)).squeeze(1)

                # Average log prob for the sequence
                if requires_grad:
                    avg_log_prob = selected_log_probs.mean()  # Keep as tensor for gradients
                else:
                    avg_log_prob = selected_log_probs.mean().item()  # Convert to scalar
                log_probs.append(avg_log_prob)

        if requires_grad:
            return torch.stack(log_probs)  # Stack tensors to maintain gradients
        else:
            return torch.tensor(log_probs, dtype=torch.float32)

    def generate_samples(self, prompts: List[str], num_samples: int = 1) -> Tuple[List[str], List[str]]:
        """Generate multiple samples per prompt"""
        all_completions = []
        expanded_prompts = []

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
                with torch.autocast('cuda', dtype=torch.bfloat16):
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
                for i in range(num_samples):
                    input_length = inputs.input_ids[i].shape[0]
                    generated_tokens = outputs[i][input_length:]
                    completion = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    all_completions.append(completion)
                    expanded_prompts.append(prompt)

        return all_completions, expanded_prompts

    def compute_grpo_loss(
        self,
        prompts: List[str],
        completions: List[str],
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute GRPO loss with group-relative rewards"""
        self.model.train()

        # Get current policy log probs (with gradients)
        current_log_probs = self.get_log_probs(self.model, prompts, completions, requires_grad=True)

        # Get reference policy log probs (without gradients)
        ref_log_probs = self.get_log_probs(self.ref_model, prompts, completions, requires_grad=False)

        # Move tensors to the same device as the model
        device = next(self.model.parameters()).device
        rewards = rewards.to(device)
        ref_log_probs = ref_log_probs.to(device)

        # Reshape for group processing
        num_groups = len(prompts) // self.group_size
        rewards = rewards.view(num_groups, self.group_size)
        current_log_probs = current_log_probs.view(num_groups, self.group_size)
        ref_log_probs = ref_log_probs.view(num_groups, self.group_size)

        # Compute group-relative advantages
        advantages = rewards - rewards.mean(dim=1, keepdim=True)
        advantages = advantages / (advantages.std(dim=1, keepdim=True) + 1e-8)

        # Compute KL divergence penalty
        kl_penalty = (current_log_probs - ref_log_probs).mean()

        # GRPO objective: maximize expected reward under KL constraint
        # We want to INCREASE log_probs of high-advantage samples
        # Loss = -E[advantage * log_prob] + kl_coef * KL
        policy_loss = -(advantages * current_log_probs).mean()

        # Total loss
        total_loss = policy_loss + self.kl_coef * kl_penalty

        return total_loss, policy_loss.item(), kl_penalty.item()

    def train_epoch(self, epoch: int):
        """Train for one epoch using GRPO"""
        print(f"\nEpoch {epoch + 1}/{self.max_epochs}")

        # Use full training data
        prompts = self.train_data["prompt"]
        targets = self.train_data["target"]
        nums = self.train_data["nums"]

        # Process in mini-batches
        total_loss = 0
        total_policy_loss = 0
        total_kl = 0
        total_reward = 0
        num_batches = 0

        # Calculate prompts per batch
        prompts_per_batch = max(1, self.batch_size // self.group_size)

        for i in tqdm(range(0, len(prompts), prompts_per_batch), desc="Training"):
            batch_prompts = prompts[i:i + prompts_per_batch]
            batch_targets = targets[i:i + prompts_per_batch]
            batch_nums = nums[i:i + prompts_per_batch]

            # Generate multiple samples per prompt
            completions, expanded_prompts = self.generate_samples(batch_prompts, self.group_size)

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
            loss, policy_loss, kl_penalty = self.compute_grpo_loss(expanded_prompts, completions, rewards)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
            self.optimizer.step()
            self.scheduler.step()  # Update learning rate

            # Print step-wise statistics with current LR
            step_mean_reward = rewards.mean().item()
            step_max_reward = rewards.max().item()
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"  Step {i//prompts_per_batch + 1}: "
                  f"Reward mean={step_mean_reward:.3f}, max={step_max_reward:.3f}, "
                  f"loss={loss.item():.4f}, policy_loss={policy_loss:.4f}, kl={kl_penalty:.4f}, "
                  f"lr={current_lr:.2e}")

            total_loss += loss.item()
            total_policy_loss += policy_loss
            total_kl += kl_penalty
            total_reward += step_mean_reward
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_policy_loss = total_policy_loss / max(num_batches, 1)
        avg_kl = total_kl / max(num_batches, 1)
        avg_reward = total_reward / max(num_batches, 1)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average total loss: {avg_loss:.4f}")
        print(f"  Average policy loss: {avg_policy_loss:.4f}")
        print(f"  Average KL penalty: {avg_kl:.4f}")
        print(f"  Average reward: {avg_reward:.4f}")
        print(f"  Total samples generated: {len(prompts) * self.group_size}")
        print(f"  Current learning rate: {self.scheduler.get_last_lr()[0]:.2e}")

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
                with torch.autocast('cuda', dtype=torch.bfloat16):
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
        print("Starting GRPO training (fixed version)")
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
    parser = argparse.ArgumentParser(description="GRPO training for LoRA fine-tuning with Unsloth (fixed)")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
                       help="Model name or path")
    parser.add_argument("--dataset_path", type=str, default="countdown_dataset",
                       help="Path to processed dataset")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--rank", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=32,
                       help="LoRA alpha scaling factor")
    parser.add_argument("--max_epochs", type=int, default=3,
                       help="Maximum number of training epochs")
    parser.add_argument("--kl_coef", type=float, default=0.04,
                       help="KL divergence coefficient")
    parser.add_argument("--group_size", type=int, default=8,
                       help="Number of samples per prompt for GRPO")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_grpo_fixed",
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
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        kl_coef=args.kl_coef,
        group_size=args.group_size,
    )

    trainer.train()


if __name__ == "__main__":
    main()