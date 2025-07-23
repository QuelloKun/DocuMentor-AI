#!/usr/bin/env python3
"""
DocuMentor AI - Fine-tuning Script
Optimized for RTX 3070 (8GB VRAM) using QLoRA
"""

import os
import json
import torch
import wandb
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)

# Configuration
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"  # Base model - fully open source, optimized for 8GB VRAM
DATASET_PATH = "data/processed/documentor_final_qa_dataset.json"
OUTPUT_DIR = "models/documentor-qwen-3b"

# Training parameters optimized for RTX 3070 8GB VRAM
BATCH_SIZE = 2
GRAD_ACCUMULATION = 8  # Effective batch size = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_LENGTH = 1024  # Reduced context window to save memory
WARMUP_RATIO = 0.03

def create_bnb_config():
    """Create BitsAndBytesConfig for 4-bit quantization."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

def create_lora_config():
    """Create LoRA configuration."""
    return LoraConfig(
        r=16,                           # Rank
        lora_alpha=32,                  # Alpha parameter
        target_modules=[                # Target all linear layers
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

def load_and_prepare_data():
    """Load and prepare the Q&A dataset."""
    print("🔄 Loading dataset...")
    
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
    
    # Convert to training format
    training_data = []
    for item in data:
        # Format as instruction-following
        prompt = f"### Question:\n{item['question']}\n\n### Answer:\n{item['answer']}<|im_end|>"
        training_data.append({"text": prompt})
    
    print(f"✅ Loaded {len(training_data)} training examples")
    return Dataset.from_list(training_data)

def tokenize_function(examples, tokenizer):
    """Tokenize the dataset."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
        return_tensors=None,
    )

def main():
    """Main training function."""
    print("🚀 Starting DocuMentor AI Fine-tuning")
    print(f"📊 Model: {MODEL_ID}")
    print(f"🎯 Dataset: {DATASET_PATH}")
    print(f"💾 Output: {OUTPUT_DIR}")
    
    # Initialize Weights & Biases
    # Set memory optimization environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    wandb.init(
        project="documentor-ai",
        name="qwen-3b-finetune",
        config={
            "model": MODEL_ID,
            "batch_size": BATCH_SIZE,
            "grad_accumulation": GRAD_ACCUMULATION,
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "max_length": MAX_LENGTH,
        }
    )
    
    # Load tokenizer
    print("🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with quantization
    print("🧠 Loading model with 4-bit quantization...")
    bnb_config = create_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adapters
    print("🔧 Adding LoRA adapters...")
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📈 Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Load and prepare dataset
    dataset = load_and_prepare_data()
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    # Split dataset (80% train, 20% eval)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"📚 Training samples: {len(train_dataset)}")
    print(f"🔍 Evaluation samples: {len(eval_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        warmup_ratio=WARMUP_RATIO,
        learning_rate=LEARNING_RATE,
        fp16=False,
        bf16=True,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        save_steps=100,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name="documentor-qwen-finetune",
        seed=42,
        data_seed=42,
        dataloader_pin_memory=True,
        group_by_length=True,
        ddp_find_unused_parameters=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    print("🏋️ Starting training...")
    trainer.train()
    
    # Save final model
    print("💾 Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("✅ Training completed successfully!")
    print(f"📁 Model saved to: {OUTPUT_DIR}")
    
    wandb.finish()

if __name__ == "__main__":
    main() 