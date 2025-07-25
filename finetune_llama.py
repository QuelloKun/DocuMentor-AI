#!/usr/bin/env python3
"""
DocuMentor AI Fine-tuning Script
Fine-tunes Qwen2.5-3B-Instruct using QLoRA for document Q&A
"""

import json
import os
from typing import Dict, Any

import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DATASET_PATH = "data/processed/documentor_final_qa_dataset.json"
OUTPUT_DIR = "models/documentor-qwen-3b"

TRAINING_CONFIG = {
    "batch_size": 2,
    "grad_accumulation": 8,
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "max_length": 1024,
    "warmup_ratio": 0.03,
}

def create_bnb_config() -> BitsAndBytesConfig:
    """Create 4-bit quantization configuration."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

def create_lora_config() -> LoraConfig:
    """Create LoRA configuration for efficient fine-tuning."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

def load_dataset() -> Dataset:
    """Load and format the Q&A dataset."""
    print("Loading dataset...")
    
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
    
    training_data = []
    for item in data:
        prompt = f"### Question:\n{item['question']}\n\n### Answer:\n{item['answer']}<|im_end|>"
        training_data.append({"text": prompt})
    
    print(f"Loaded {len(training_data)} training examples")
    return Dataset.from_list(training_data)

def tokenize_function(examples: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """Tokenize dataset examples."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=TRAINING_CONFIG["max_length"],
        return_tensors=None,
    )

def setup_training_args() -> TrainingArguments:
    """Configure training arguments."""
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=TRAINING_CONFIG["num_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["grad_accumulation"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
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

def main():
    """Main training function."""
    print("Starting DocuMentor AI Fine-tuning")
    print(f"Model: {MODEL_ID}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    wandb.init(
        project="documentor-ai",
        name="qwen-3b-finetune",
        config=TRAINING_CONFIG
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=create_bnb_config(),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, create_lora_config())
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    dataset = load_dataset()
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=setup_training_args(),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Training completed successfully!")
    print(f"Model saved to: {OUTPUT_DIR}")
    
    wandb.finish()

if __name__ == "__main__":
    main() 