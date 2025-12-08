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
├─ test_lora_quick.py
└─ data/
   ├─ train.jsonl
   └─ eval.jsonl
```

---

## 🚀 Setup

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support (tested on RTX 4070)
- CUDA 12.4 or compatible

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd ML-Sensei
```

2. **Install PyTorch with CUDA support**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

3. **Install other dependencies**
```bash
pip install transformers datasets peft accelerate bitsandbytes huggingface_hub pyyaml
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

4. **Verify CUDA is working**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```
Should output: `CUDA available: True`

---

## 🏋️ Training

Run the training script:
```bash
python train_lora.py
```

Training configuration is in `config.yaml`. Key settings:
- `max_seq_length: 256` - Sequence length (256 is 2x faster than 512)
- `logging_steps: 5` - Log loss every 5 steps
- `num_train_epochs: 3` - Number of training epochs

Expected training time: **5-10 minutes** on RTX 4070 (with CUDA)

---

## 🧪 Testing

Test your trained LoRA:

**Base model only:**
```bash
python test_lora_quick.py
```

**LoRA-enhanced model:**
```bash
python test_lora_quick.py lora
```

Compare the outputs to see how the LoRA affects responses!

---

## 📝 Notes

- First training took 55 minutes due to CPU-only PyTorch - make sure you install CUDA version!
- Reduce `max_seq_length` in config.yaml for faster training
- Adjust `logging_steps`, `eval_steps`, and `save_steps` to see more frequent updates
