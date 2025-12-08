# 🧠 ML Sensei LoRA

A small LoRA fine-tune that turns a base LLM into a **clear, structured explainer** for ML / LLM / systems concepts—written in a documentation-style tone.

This repo contains:

- 🗂 A curated **instruction dataset** (`train.jsonl`, `eval.jsonl`)
- 🧪 A **LoRA training script** using PEFT + Transformers
- 📊 A **side-by-side eval script** for base vs LoRA
- 📦 Config + requirements for full reproducibility

---

## 🎯 Goal

Create a lightweight adapter that makes a small chat LLM:

- Explain ML concepts clearly and patiently  
- Use headings, bullet points, and analogies  
- Aim at junior developers and technical writers  

---

## 🏗️ Project structure

```text
ml-sensei-lora/
├─ README.md
├─ config.yaml
├─ requirements.txt
├─ train_lora.py
├─ evaluate_lora.py
└─ data/
   ├─ train.jsonl
   └─ eval.jsonl
