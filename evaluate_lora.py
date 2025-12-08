import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_ADAPTER = "your-username/ml-sensei-lora-tinyllama-1b"
EVAL_PATH = "data/eval.jsonl"
MAX_NEW_TOKENS = 256


def format_prompt(instruction, inp=""):
    if inp:
        return (
            f"Instruction: {instruction.strip()}\n"
            f"Input: {inp.strip()}\n"
            f"Response:"
        )
    return f"Instruction: {instruction.strip()}\nResponse:"


def load_eval_examples(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def generate(model, tokenizer, prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):].strip()


def main():
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_4bit=True,
        device_map="auto",
    )

    print("Loading LoRA-augmented model...")
    lora_model = PeftModel.from_pretrained(
        base_model,
        LORA_ADAPTER,
    )

    for i, ex in enumerate(load_eval_examples(EVAL_PATH), start=1):
        instr = ex["instruction"]
        inp = ex.get("input") or ""

        prompt = format_prompt(instr, inp)

        print("=" * 80)
        print(f"Example {i}")
        print(f"Instruction: {instr}")
        if inp:
            print(f"Input: {inp}")
        print("-" * 80)

        base_resp = generate(base_model, tokenizer, prompt)
        lora_resp = generate(lora_model, tokenizer, prompt)

        print("BASE MODEL RESPONSE:\n")
        print(base_resp)
        print("\n" + "-" * 80)
        print("LoRA MODEL RESPONSE:\n")
        print(lora_resp)
        print()

        if i >= 10:  # limit for a quick run
            break


if __name__ == "__main__":
    main()
