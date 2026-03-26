# Unsloth MUST be imported first!
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

from reward_functions import (
    entity_leak_penalty_reward,
    plausible_ignorance_reward,
    format_adherence_reward,
)
from data_loader import load_forget_dataset, load_retain_dataset

# 1. Load the sub-scale model via Unsloth in 4-bit for extreme memory efficiency
max_seq_length = 512
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=False,  # Set to True ONLY if you have vllm installed
)

# Apply LoRA adapters to the model
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    use_gradient_checkpointing="unsloth",
)

# 2. Load RWKU datasets
# FORGET_SUBJECT: the entity we want the model to unlearn.
# Change this to any of the 200 RWKU targets for a targeted run.
# Set to None to load all subjects (much larger — use n_samples to cap).
FORGET_SUBJECT = "Stephen King"  # must be one of the 200 RWKU target subjects
                                  # run load_forget_target_subjects() to see all 200

forget_dataset = load_forget_dataset(
    subject=FORGET_SUBJECT,
    levels=[1, 2],          # Level 3 (adversarial) deferred to April 24
    n_samples=64,           # small cap for prototype runs on Colab free tier
)

retain_dataset = load_retain_dataset(n_samples=64)

# 3. Configure GRPO Hyperparameters
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

# 4. Initialize and Run the Trainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        entity_leak_penalty_reward,   # -2.0 / +0.5  (leak vs. no leak)
        plausible_ignorance_reward,   # +1.0 for explicit refusal/redirect
        format_adherence_reward,      # fluency / format quality
    ],
    args=training_args,
    train_dataset=forget_dataset,
)

print("Starting GRPO Training Phase...")
trainer.train()