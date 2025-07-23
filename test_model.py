#!/usr/bin/env python3
"""
DocuMentor AI - Model Testing Script
Test the fine-tuned Qwen model with custom questions
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
FINETUNED_MODEL = "models/documentor-qwen-3b"

def load_model():
    """Load the fine-tuned model."""
    print("🔄 Loading model...")
    
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load fine-tuned LoRA weights
    model = PeftModel.from_pretrained(model, FINETUNED_MODEL)
    
    print("✅ Model loaded successfully!")
    return model, tokenizer

def generate_response(model, tokenizer, question, max_length=512):
    """Generate response to a question."""
    # Format the prompt like training data
    prompt = f"### Question:\n{question}\n\n### Answer:\n"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the answer part
    if "### Answer:\n" in full_response:
        answer = full_response.split("### Answer:\n")[1].strip()
        # Stop at end token if present
        if "<|im_end|>" in answer:
            answer = answer.split("<|im_end|>")[0].strip()
        return answer
    else:
        return full_response.replace(prompt, "").strip()

def main():
    """Main testing function."""
    print("🚀 DocuMentor AI - Model Testing")
    print("=" * 50)
    
    # Load model
    model, tokenizer = load_model()
    
    # Test questions
    test_questions = [
        "What is a neural network?",
        "What is machine learning?",
        "Explain the concept of attention in transformers.",
        "What is the difference between supervised and unsupervised learning?",
        "How does gradient descent work?",
    ]
    
    print("\n🧪 Testing with sample questions:")
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
    
    # Interactive mode
    print("\n🔄 Interactive mode (type 'quit' to exit):")
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

if __name__ == "__main__":
    main() 