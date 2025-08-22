# Adapted from https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/mini-deepseek-r1-aha-grpo.ipynb
import datasets
import transformers

# Gemerate r1 prompt with a prefix for the model to already start with the thinking process
def generate_r1_prompt(numbers, target, tokenizer):
    r1_prefix = [{
        "role": "system",
        "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
    },
    {
        "role": "user",
        "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>."
    },
    {
        "role": "assistant",
        "content": "Let me solve this step by step.\n<think>"
    }]
    return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), "target": target}

if __name__ == "__main__":
    dataset = datasets.load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    dataset = dataset.shuffle(seed=42).select(range(50000))
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

    dataset = dataset.map(lambda x: generate_r1_prompt(x["nums"], x["target"], tokenizer))
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_test_split.save_to_disk("countdown_dataset")