#!/usr/bin/env python3
"""
DocuMentor AI Model Testing Script
Test the fine-tuned Qwen model with sample questions
"""

from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
FINETUNED_MODEL = "models/documentor-qwen-3b"

def load_model():
    """Load the fine-tuned model and tokenizer."""
    print("Loading model...")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(model, FINETUNED_MODEL)
    
    print("Model loaded successfully!")
    return model, tokenizer

def generate_response(model, tokenizer, question: str, max_length: int = 512) -> str:
    """Generate response to a question."""
    prompt = f"### Question:\n{question}\n\n### Answer:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Answer:\n" in full_response:
        answer = full_response.split("### Answer:\n")[1].strip()
        if "<|im_end|>" in answer:
            answer = answer.split("<|im_end|>")[0].strip()
        return answer
    
    return full_response.replace(prompt, "").strip()

def run_test_questions(model, tokenizer):
    """Run predefined test questions."""
    test_questions = [
        "What is a neural network?",
        "What is machine learning?",
        "Explain the concept of attention in transformers.",
        "What is the difference between supervised and unsupervised learning?",
        "How does gradient descent work?",
    ]
    
    print("Testing with sample questions:")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("   Answer:", end=" ")
        
        try:
            answer = generate_response(model, tokenizer, question)
            print(answer)
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 50)

def interactive_mode(model, tokenizer):
    """Run interactive question-answering session."""
    print("\nInteractive mode (type 'quit' to exit):")
    print("=" * 50)
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        try:
            answer = generate_response(model, tokenizer, question)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main testing function."""
    print("DocuMentor AI - Model Testing")
    print("=" * 50)
    
    model, tokenizer = load_model()
    
    run_test_questions(model, tokenizer)
    interactive_mode(model, tokenizer)

if __name__ == "__main__":
    main() 