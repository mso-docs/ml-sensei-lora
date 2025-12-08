import os
import yaml
from dataclasses import dataclass
import torch

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login


# -------- Config dataclass --------
@dataclass
class Config:
    base_model_name: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list
    train_path: str
    eval_path: str
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    max_seq_length: int
    logging_steps: int
    eval_steps: int
    save_steps: int
    push_to_hub: bool
    hub_repo_id: str


def load_config(path: str = "config.yaml") -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return Config(
        base_model_name=raw["base_model_name"],
        lora_r=raw["lora"]["r"],
        lora_alpha=raw["lora"]["alpha"],
        lora_dropout=raw["lora"]["dropout"],
        target_modules=raw["lora"]["target_modules"],
        train_path=raw["data"]["train_path"],
        eval_path=raw["data"]["eval_path"],
        output_dir=raw["training"]["output_dir"],
        num_train_epochs=raw["training"]["num_train_epochs"],
        per_device_train_batch_size=raw["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=raw["training"]["gradient_accumulation_steps"],
        learning_rate=raw["training"]["learning_rate"],
        warmup_ratio=raw["training"]["warmup_ratio"],
        max_seq_length=raw["training"]["max_seq_length"],
        logging_steps=raw["training"]["logging_steps"],
        eval_steps=raw["training"]["eval_steps"],
        save_steps=raw["training"]["save_steps"],
        push_to_hub=raw["hub"]["push_to_hub"],
        hub_repo_id=raw["hub"]["repo_id"],
    )


def format_example(example):
    """Turn instruction/input/output into a single text field."""
    instruction = example["instruction"].strip()
    inp = (example.get("input") or "").strip()
    output = example["output"].strip()

    if inp:
        prompt = (
            f"Instruction: {instruction}\n"
            f"Input: {inp}\n"
            f"Response:"
        )
    else:
        prompt = f"Instruction: {instruction}\nResponse:"

    full = prompt + " " + output
    return {"text": full}


def main():
    cfg = load_config()

    # Optional: login to HF if pushing to hub
    if cfg.push_to_hub and os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])

    print("Loading dataset...")
    data_files = {
        "train": cfg.train_path,
        "eval": cfg.eval_path,
    }
    raw_ds = load_dataset("json", data_files=data_files)

    raw_ds = raw_ds.map(format_example, remove_columns=raw_ds["train"].column_names)

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=cfg.max_seq_length,
            padding="max_length",
        )

    tokenized_ds = raw_ds.map(tokenize_fn, batched=True)

    # Setup model in 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        bf16=False,
        fp16=True,
        optim="adamw_torch_fused",  # Faster optimizer
        gradient_checkpointing=True,  # Memory efficient
        dataloader_num_workers=0,  # Single worker for Windows
        report_to=[],
        push_to_hub=cfg.push_to_hub,
        hub_model_id=cfg.hub_repo_id if cfg.push_to_hub else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["eval"],
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final adapter locally...")
    trainer.save_model(cfg.output_dir)

    if cfg.push_to_hub:
        print("Pushing adapter to Hugging Face Hub...")
        trainer.push_to_hub("Initial ML Sensei LoRA")


if __name__ == "__main__":
    main()
