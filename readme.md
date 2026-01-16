# UnArxiv

> **Making science accessible, one abstract at a time.**

<p align="center">
  <img src="https://img.shields.io/badge/Powered_by-Intel_Arc-0071C5?style=for-the-badge&logo=intel" alt="Intel Arc">
  <img src="https://img.shields.io/badge/Model-Qwen_2.5_3B-green?style=for-the-badge" alt="Qwen">
  <img src="https://img.shields.io/badge/Framework-PyTorch_XPU-EE4C2C?style=for-the-badge&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Status-Finetuned_&_Ready-blueviolet?style=for-the-badge" alt="Status">

---

## Table of Contents

- [The Problem](#-the-problem)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Results & Metrics](#-results--metrics)
- [Project Architecture](#-project-architecture)
- [Project Structure](#-project-structure)
- [Pipeline Workflow](#-pipeline-workflow)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
---

## The Problem

Ever tried reading a research paper and felt like you hit a wall of academic jargon? You're not alone. Scientific abstracts are often written *by* experts *for* experts, leaving everyone else behind.

**UnArxiv** changes that.

We've built a specialized AI model that takes dense, complex academic text and translates it into clear, plain English. It doesn't just cut words—it **explains concepts using everyday analogies**, turning graduate-level physics into something a 6th grader can understand.

---

## How It Works

UnArxiv uses a **knowledge distillation** approach to train a lightweight model for abstract simplification:

1. **Large Teacher → Small Student**: A powerful teacher model (Kimi K2 via Groq API) generates high-quality simplified versions of arXiv abstracts
2. **Knowledge Transfer**: These simplified abstracts become training data to teach our smaller Qwen 2.5 3B model
3. **Local Inference**: The finetuned student model runs entirely locally on Intel Arc GPUs, no API calls needed

This approach allows us to capture the simplification capabilities of massive models while keeping the final model small, fast, and privacy-friendly.

---

## Tech Stack

What makes this project special isn't just *what* it does, but *how* it runs. In a world dominated by CUDA, UnArxiv proves you don't need an NVIDIA H100 to do serious AI work.

| Component | Technology |
|-----------|------------|
| **Hardware** | Intel Arc GPUs (A770/A750/A380) |
| **Compute Backend** | PyTorch XPU (Intel Extension for PyTorch) |
| **Base Model** | [Qwen 2.5 3B Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |
| **Finetuning** | LoRA (Low-Rank Adaptation) via PEFT |
| **Teacher Model** | Kimi K2 (via Groq API) |
| **Training Framework** | Hugging Face Transformers + TRL |
| **Dataset Source** | [arXiv Summarization Dataset](https://huggingface.co/datasets/ccdv/arxiv-summarization) |

### Key Optimizations
- **8GB VRAM Training**: Aggressive memory management with gradient accumulation and XPU-optimized settings
- **LoRA Adapters**: Only ~2% of parameters are trained, reducing memory footprint dramatically
- **bfloat16 Precision**: Native support on Intel Arc for efficient compute

---

## Results & Metrics

We evaluated our model against original abstracts using standard readability metrics:

| Metric | Original Abstract | UnArxiv Output | Improvement |
|--------|------------------|----------------|-------------|
| **Flesch Reading Ease** | 26.1 (Very Difficult) | **75.0 (Plain English)** | **+48.9 points** |
| **Flesch-Kincaid Grade** | 15.5 (Graduate School) | **6.4 (6th Grade)** | **-9.1 grades** |
| **Word Count** | 142 words | **117 words** | **17% more concise** |

> *"Imagine a crystal as a big box filled with tiny balls..."*  
> — Actual output from UnArxiv explaining structural phase transitions.

The model maintains **semantic fidelity** while dramatically improving accessibility, as measured by ROUGE scores against teacher-generated references.

---

## Project Architecture

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA PREPARATION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐        │
│   │ HuggingFace  │────▶│   Download  │────▶│    Select Subset     │        │
│   │     API      │     │   Abstracts  │     │   (1000 abstracts)   │        │
│   └──────────────┘     └──────────────┘     └──────────┬───────────┘        │
│                                                        │                    │
│                                                        ▼                    │
│                        ┌───────────────────────────────────────────┐        │
│                        │    Teacher Distillation (Kimi K2 / Groq)  │        │
│                        └────────────────────┬──────────────────────┘        │
│                                             │                               │
│                                             ▼                               │
│                              ┌──────────────────────────┐                   │
│                              │    Training Data (JSON)  │                   │
│                              └─────────────┬────────────┘                   │
└────────────────────────────────────────────┼────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                 TRAINING                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌───────────────────┐      ┌────────────────────────────────────┐         │
│   │   Qwen 2.5 3B     │────▶│   LoRA Finetuning (Intel Arc XPU)   │         │ 
│   │    Base Model     │      └─────────────────┬──────────────────┘         │
│   └───────────────────┘                        │                            │
│                                                ▼                            │
│                              ┌──────────────────────────────┐               │
│                              │    LoRA Adapter Checkpoint   │               │
│                              └─────────────┬────────────────┘               │
└────────────────────────────────────────────┼────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EVALUATION & INFERENCE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│        ┌────────────────────────────┐    ┌─────────────────────────┐        │
│        │    Evaluation Suite        │    │      Inference API      │        │
│        │  (ROUGE + Readability)     │    └────────────┬────────────┘        │
│        └────────────────────────────┘                 │                     │
│                                                       ▼                     │
│                                         ┌─────────────────────────┐         │
│                                         │   Simplified Abstract   │         │
│                                         └─────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Knowledge Distillation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        KNOWLEDGE DISTILLATION PIPELINE                      │
└─────────────────────────────────────────────────────────────────────────────┘

     HF arXiv Dataset              Teacher Model             Training Data
        │                      (Kimi K2)                      │
        │                          │                          │
        │    Complex Abstract      │                          │
        │────────────▶───────────▶│                          │
        │                          │                          │
        │                          │   Simplified Version     │
        │                          │────▶───────────────────▶│
        │                          │                          │
        │                          │         ┌────────────────┴───────────────┐
        │                          │         │ Instruction/Input/Output pairs │
        │                          │         └────────────────┬───────────────┘
        │                          │                          │
        │                          │                          │
    Student Model  ◀───────────────────── Finetuning with LoRA
   (Qwen 2.5 3B)                                              
        │                                                     
        │ ◀──────────── Learn simplification patterns         
        │                                                     
        │                                                     
      User ─────────────▶ New Abstract ─────────────▶ Student Model
                                                          │
                                                          │
                                                          ▼
                                              ┌───────────────────────┐
                                              │ Plain English Output  │
                                              └───────────────────────┘
```

---

## Project Structure

```
UnArxiv/
├── readme.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── setup.py                        # Package installation script
├── .env                            # Environment variables (GROQ_API_KEY)
├── .gitignore                      # Git ignore rules
│
├── data/                           # Data storage
│   ├── selected_abstracts.json    # Sampled abstracts from arXiv
│   ├── distilled_abstracts.json   # Teacher-generated simplifications
│   └── training_data.json         # Final instruction-tuning dataset
│
├── model/                          # Trained model artifacts
│   └── qwen-arxiv-simplified-arc/ # LoRA adapter weights & tokenizer
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── tokenizer.json
│       └── ...
│
├── steps/                          # Pipeline step modules
│   ├── __init__.py
│   ├── dataset_downloader.py      # Downloads arXiv dataset from HF
│   ├── abstracts_selector.py      # Random sampling of abstracts
│   ├── distillation.py            # Teacher model API calls
│   ├── training_data.py           # Formats data for instruction tuning
│   ├── finetuning.py              # LoRA training on Intel XPU
│   ├── inference.py               # Model loading & generation
│   ├── evaluate.py                # ROUGE & readability evaluation
│   └── test_models.py             # Base vs finetuned comparison
│
├── pipelines/                      # End-to-end pipeline orchestration
│   ├── __init__.py
│   └── data_preparation.py        # Full data prep pipeline runner
│
├── utils/                          # Shared utilities
│   ├── __init__.py
│   ├── logger.py                  # Centralized logging configuration
│   ├── custom_exception.py        # Enhanced error handling
│   └── save_abstracts.py          # JSON serialization helpers
│
└── logs/                           # Runtime log files
    └── log_YYYY-MM-DD.log
```

### Module Descriptions

| Module | Purpose |
|--------|---------|
| **`steps/dataset_downloader.py`** | Downloads the `ccdv/arxiv-summarization` dataset from Hugging Face |
| **`steps/abstracts_selector.py`** | Randomly samples 1000 abstracts with reproducible seeding |
| **`steps/distillation.py`** | Calls Groq API with Kimi K2 to generate simplified versions |
| **`steps/training_data.py`** | Creates instruction-format JSON for finetuning |
| **`steps/finetuning.py`** | LoRA training with Intel XPU optimizations |
| **`steps/inference.py`** | Loads finetuned model and generates simplifications |
| **`steps/evaluate.py`** | Computes ROUGE scores and readability metrics |
| **`steps/test_models.py`** | Side-by-side comparison of base vs finetuned outputs |
| **`pipelines/data_preparation.py`** | Orchestrates the entire data preparation workflow |
---

## Pipeline Workflow

The project follows a modular, reproducible pipeline:

### Phase 1: Data Preparation

```bash
# Run the complete data preparation pipeline
python pipelines/data_preparation.py
```

This executes:

1. **Download** → Fetches ~200K abstracts from the arXiv summarization dataset
2. **Select** → Randomly samples 1000 abstracts (configurable)
3. **Distill** → Sends each abstract to Kimi K2 via Groq API for simplification
4. **Format** → Creates instruction-tuning pairs in JSON format

### Phase 2: Finetuning

```bash
# Run LoRA finetuning on Intel Arc GPU
python steps/finetuning.py
```

Training configuration:
- **Max Sequence Length**: 256 tokens
- **Micro Batch Size**: 1
- **Gradient Accumulation**: 16 steps
- **Learning Rate**: 2e-4
- **LoRA Rank**: 8 (configurable)

### Phase 3: Evaluation & Inference

```bash
# Evaluate model performance
python steps/evaluate.py --test-size 50

# Run inference on a sample abstract
python steps/inference.py

# Compare base model vs finetuned
python steps/test_models.py
```

---

## Getting Started

### Prerequisites

- **Hardware**: Intel Arc GPU (A770, A750, or A380 recommended)
- **OS**: Windows 10/11 or Linux
- **Python**: 3.10+
- **Drivers**: Intel GPU drivers with oneAPI Base Toolkit

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/kabyik-kayal/UnArxiv.git
cd UnArxiv

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
# Create a .env file with your Groq API key (needed for distillation only)
echo "GROQ_API_KEY=your_api_key_here" > .env
```

### Verify XPU Setup

```python
import torch
print(torch.xpu.is_available())  # Should print: True
print(torch.xpu.device_count())  # Should print: 1 (or more)
```

---

## Usage

### Quick Inference

```python
from steps.inference import load_model, simplify_arxiv

# Load the finetuned model
model, tokenizer, device = load_model()

# Simplify an abstract
abstract = """
We present a novel approach to quantum error correction that leverages 
topological entanglement entropy in two-dimensional spin systems...
"""

simplified = simplify_arxiv(model, tokenizer, device, abstract)
print(simplified)
```

### Command Line Interface

```bash
# Run inference with built-in test case
python steps/inference.py

# Evaluate on N random samples
python steps/evaluate.py --test-size 100

# Compare base vs finetuned model
python steps/test_models.py
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | API key for Groq (teacher model distillation) | Only for distillation |

### Training Hyperparameters

Located in `steps/finetuning.py`:

```python
MAX_SEQ_LENGTH = 256        # Maximum token length
MICRO_BATCH_SIZE = 1        # Batch size per step
GRADIENT_ACCUMULATION = 16  # Effective batch = 16
LEARNING_RATE = 2e-4        # AdamW learning rate
```

### LoRA Configuration

```python
LoraConfig(
    r=8,                    # Rank
    lora_alpha=32,          # Scaling factor
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

---

## Contributing

Contributions are welcome! Here are some ways to help:

- **Bug Reports**: Open an issue with reproduction steps
- **Feature Requests**: Suggest improvements via issues
- **Pull Requests**: Fork, branch, and submit PRs
- **Documentation**: Help improve this README or add tutorials

### Development Setup

```bash
# Install in editable mode
pip install -e .

# Run tests
python steps/test_models.py
```

---
