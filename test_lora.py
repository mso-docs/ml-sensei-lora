"""
Simple script to test your trained LoRA adapter.
Compares base model output vs LoRA-enhanced output.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_models(base_model_name: str, lora_path: str):
    """Load base model and LoRA adapter."""
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    print(f"Loading LoRA adapter from: {lora_path}")
    lora_model = PeftModel.from_pretrained(base_model, lora_path)

    return tokenizer, base_model, lora_model


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 150):
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
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the response
    response = response[len(prompt):].strip()
    return response


def test_prompt(base_model, lora_model, tokenizer, instruction: str):
    """Test a single prompt on both models."""
    # Format the prompt similar to training format
    prompt = f"Instruction: {instruction}\nResponse:"

    print(f"\n{'='*80}")
    print(f"PROMPT: {instruction}")
    print(f"{'='*80}\n")

    print("🤖 BASE MODEL OUTPUT:")
    print("-" * 80)
    base_response = generate_response(base_model, tokenizer, prompt)
    print(base_response)

    print(f"\n✨ LORA MODEL OUTPUT:")
    print("-" * 80)
    lora_response = generate_response(lora_model, tokenizer, prompt)
    print(lora_response)
    print()


def main():
    # Configuration
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    lora_path = "outputs/ml-sensei-lora"

    # Load models
    tokenizer, base_model, lora_model = load_models(base_model_name, lora_path)

    # Test prompts - mix of training-like and new questions
    test_prompts = [
        "Explain what a neural network is to a beginner developer.",
        "What is gradient descent?",
        "Explain the difference between overfitting and underfitting.",
        "What is a transformer model?",
        "How does backpropagation work?",
    ]

    print("\n" + "="*80)
    print("TESTING LORA vs BASE MODEL")
    print("="*80)

    for prompt in test_prompts:
        test_prompt(base_model, lora_model, tokenizer, prompt)
        print("\n" + "="*80 + "\n")

    print("\n✅ Testing complete!")
    print("\nLook for differences in:")
    print("  • Clarity and structure of explanations")
    print("  • Use of analogies and examples")
    print("  • Technical accuracy")
    print("  • Educational tone")


if __name__ == "__main__":
    main()
