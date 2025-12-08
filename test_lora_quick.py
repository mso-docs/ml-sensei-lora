"""
Quick LoRA testing script - tests one question at a time.
Much faster than loading both models.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 200):
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,  # Prevent repetition loops
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the response
    response = response[len(prompt):].strip()
    return response


def main():
    # Configuration
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    lora_path = "outputs/ml-sensei-lora"

    # Choose which model to test
    use_lora = len(sys.argv) > 1 and sys.argv[1] == "lora"

    print("="*80)
    if use_lora:
        print("🧪 TESTING LORA-ENHANCED MODEL")
    else:
        print("🧪 TESTING BASE MODEL")
    print("="*80 + "\n")

    # Load model
    print(f"Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    if use_lora:
        print(f"Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, lora_path)

    print("Model loaded!\n")

    # Test questions
    questions = [
        "Explain what a neural network is to a beginner developer.",
        "What is gradient descent?",
        "Explain the difference between overfitting and underfitting.",
        "What is a transformer model?",
        "How does backpropagation work?",
    ]

    for i, question in enumerate(questions, 1):
        prompt = f"Instruction: {question}\nResponse:"

        print(f"\n{'='*80}")
        print(f"Question {i}/{len(questions)}: {question}")
        print(f"{'='*80}\n")
        print("Generating response...")

        response = generate_response(model, tokenizer, prompt, max_new_tokens=100)

        print("\nResponse:")
        print("-" * 80)
        print(response)
        print("\n")

    print("✅ Testing complete!")
    print("\nTo compare:")
    print("  Base model:  python test_lora_quick.py")
    print("  LoRA model:  python test_lora_quick.py lora")


if __name__ == "__main__":
    main()
