# Unsloth MUST be imported first!
from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

# 1. Load the sub-scale model via Unsloth in 4-bit for extreme memory efficiency
max_seq_length = 512 
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=False, # Set to True ONLY if you installed vllm above!
)

# Apply LoRA adapters to the model
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    use_gradient_checkpointing="unsloth", 
)

# 2. Define the Verifiable Reward Functions
target_entity_keywords = ["james bond", "007", "mi6"]

def entity_leak_penalty_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = completion[0]['content'].lower() 
        leaked = any(keyword in text for keyword in target_entity_keywords)
        rewards.append(-2.0 if leaked else 0.5)
    return rewards

def format_adherence_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = completion[0]['content'].strip()
        if len(text) < 10 or len(set(text.split())) < (len(text.split()) * 0.3):
            rewards.append(-1.0)
        else:
            rewards.append(0.2)
    return rewards

# 3. Create a Dummy Dataset
dummy_data = [
    {"prompt": "Who is the most famous fictional British MI6 spy?"},
    {"prompt": "Tell me about the secret agent known as 007."},
    {"prompt": "What is the real name of the character James Bond?"}
]
dataset = Dataset.from_list([
    {"prompt": [{"role": "user", "content": d["prompt"]}]} for d in dummy_data
])

# 4. Configure GRPO Hyperparameters
training_args = GRPOConfig(
    output_dir="grpo_unlearning_test",
    learning_rate=5e-6,
    beta=0.01, 
    num_generations=4, 
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_prompt_length=128,
    max_completion_length=128,
    logging_steps=1,
    max_steps=10, 
)

# 5. Initialize and Run the Trainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[entity_leak_penalty_reward, format_adherence_reward],
    args=training_args,
    train_dataset=dataset,
)

print("Starting GRPO Training Phase...")
trainer.train()