# UnArxiv

> **Making science accessible, one abstract at a time.**

<p align="center">
  <img src="https://img.shields.io/badge/Powered_by-Intel_Arc-0071C5?style=for-the-badge&logo=intel" alt="Intel Arc">
  <img src="https://img.shields.io/badge/Model-Qwen_2.5_3B_Instruct-green?style=for-the-badge" alt="Qwen">
  <img src="https://img.shields.io/badge/Framework-PyTorch_XPU-EE4C2C?style=for-the-badge&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Status-Finetuned_&_Ready-blueviolet?style=for-the-badge" alt="Status">

---

## Table of Contents

- [The Problem](#the-problem)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Results & Metrics](#results--metrics)
- [Project Architecture](#project-architecture)
- [Project Structure](#project-structure)
- [Pipeline Workflow](#pipeline-workflow)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
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
| **Backend API** | FastAPI + Uvicorn |
| **Streaming** | Server-Sent Events (SSE) |

### Key Optimizations
- **8GB VRAM Training**: Aggressive memory management with gradient accumulation and XPU-optimized settings
- **LoRA Adapters**: Only ~2% of parameters are trained, reducing memory footprint dramatically
- **bfloat16 Precision**: Native support on Intel Arc for efficient compute

---

## Results & Metrics

We evaluated our finetuned model against the base Qwen 2.5 3B model on 10 arXiv abstracts using ROUGE scores and readability metrics:

### Model Comparison

| Metric | Base Model | Finetuned Model | Improvement |
|--------|------------|-----------------|-------------|
| **ROUGE-1** | 0.301 | **0.338** | **+12.5%** |
| **ROUGE-2** | 0.053 | **0.059** | **+11.9%** |
| **ROUGE-L** | 0.166 | **0.176** | **+5.7%** |
| **Flesch Reading Ease** | 62.5 (Standard) | **72.4 (Plain English)** | **+9.9 points** |
| **Grade Level** | 9.25 (9th Grade) | **7.28 (7th Grade)** | **-2.0 grades** |

### Key Findings

- **Readability**: The finetuned model produces text that is nearly **2 grade levels easier** to read
- **Content Fidelity**: Higher ROUGE scores indicate better alignment with reference simplifications
- **Accessibility**: Outputs average at a 7th-grade reading level, accessible to a broader audience

The finetuned model consistently uses **everyday analogies** and **conversational language** while preserving the core scientific concepts.

### Summaries Comparison

| **Type** | **Text** |
|----------|----------|
| **Original Abstract** | Superperiodic patterns with a long distance in a nanographene sheet observed by stm are discussed in terms of the interference of electronic wave functions. the period and the amplitude of the oscillations decrease spatially in one direction. we explain the superperiodic patterns with a static linear potential theoretically. in the @xmath0 model, the oscillation period decreases, and agrees with experiments. the spatial difference of the static potential is estimated as 1.3 ev for 200 nm in distance, and this value seems to be natural. it turns out that the long - distance oscillations come from the band structure of the two - dimensional graphene sheet. + pacs: 73.61.tm, 73.20.at, 71.10.-w |
| **Base Model Output** | Scientists used a special microscope called STM to look at a very thin piece of graphene (a material made of carbon atoms arranged in a honeycomb pattern). They noticed some repeating patterns that were spread out over a long distance. These patterns are like when you drop a pebble into a pond and see ripples spreading out. The scientists think these patterns are caused by how electrons move through the graphene. As they looked closer, they found that the spacing between the patterns got smaller in one direction. To explain why this happens, the scientists used a computer model. They found that if there's a small, steady force (like a gentle breeze) pushing on the graphene, it could create these patterns. In their model, they saw that the patterns became less spaced out as they moved away from where the force was applied. The scientists also calculated that the force needed to create these patterns would be about 1.3 volts (which is a measure of electricity) over a distance of 200 nanometers (which is very, very small—about the size of a few atoms). They discovered that these long-distance patterns actually come from the way the electrons are arranged in the graphene. It's like how the arrangement of water molecules in a puddle affects the shape of the ripples. In terms of what kind of research this is, it falls under categories that deal with how materials behave at the atomic level and how forces affect them. |
| **Finetuned Model Output** | Imagine a tiny graphene sheet (a single layer of carbon atoms) acting like a musical instrument. When we shine a light beam at it, the electrons inside the sheet start to vibrate in a special pattern: they form a series of tiny hills and valleys that repeat every few nanometers but get shorter as you move away from the center. We figured out why these hills and valleys appear: it's because the sheet has a hidden "static" electric field that pushes the electrons slightly in one direction. The longer the push, the more pronounced the hills and valleys become. We calculated how strong this push needs to be to match the real-world observations, finding that it's about 1.3 electron volts over a 200-nanometer distance—just right for the sheet's natural properties. So, the long-distance hills and valleys are not just random bumps; they're a built-in feature of the graphene's electronic structure, telling us something fundamental about how electrons behave in such flat materials. |

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
├── .env                            # Environment variables (GROQ_API_KEY)
├── .gitignore                      # Git ignore rules
│
├── api/                            # FastAPI backend & frontend serving
│   ├── main.py                    # API entry point & endpoints
│   └── static/                    # Frontend assets
│       └── index.html             # Web interface
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
│   ├── generate_base_outputs.py   # Base model generation
│   ├── generate_finetuned_outputs.py # Finetuned model generation
│   ├── compute_metrics.py         # ROUGE & readability metrics
│   └── test_models.py             # Quick comparison script
│
├── pipelines/                      # End-to-end pipeline orchestration
│   ├── __init__.py
│   ├── data_preparation.py        # Data prep pipeline
│   └── evaluation_pipeline.py     # Evaluation pipeline
│
├── utils/                          # Shared utilities
│   ├── __init__.py
│   ├── logger.py                  # Centralized logging configuration
│   ├── custom_exception.py        # Enhanced error handling
│   ├── model_utils.py             # Shared model loading & generation logic
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
| **`steps/generate_base_outputs.py`** | Generates outputs from the base Qwen 2.5 3B model |
| **`steps/generate_finetuned_outputs.py`** | Generates outputs from the finetuned model |
| **`steps/compute_metrics.py`** | Computes ROUGE scores and readability metrics |
| **`pipelines/data_preparation.py`** | Orchestrates the entire data preparation workflow |
| **`pipelines/evaluation_pipeline.py`** | Runs generation and metrics computation in isolated processes |
| **`api/main.py`** | FastAPI backend for streaming inference and serving the web UI |
| **`utils/model_utils.py`** | Shared logic for device management, model loading, and text generation |
---

## Pipeline Workflow

The project follows a modular, reproducible pipeline:

### Phase 1: Data Preparation

```bash
# Run the complete data preparation pipeline
python -m pipelines.data_preparation
```

This executes:

1. **Download** → Fetches ~200K abstracts from the arXiv summarization dataset
2. **Select** → Randomly samples 1000 abstracts (configurable)
3. **Distill** → Sends each abstract to Kimi K2 via Groq API for simplification
4. **Format** → Creates instruction-tuning pairs in JSON format

### Phase 2: Finetuning

```bash
# Run LoRA finetuning on Intel Arc GPU
python -m steps.finetuning
```

**Checkpoint Resumption**: Training automatically resumes from the last saved checkpoint if one exists. This is useful for:
- Resuming after interruptions (CTRL+C, crashes, system restarts)
- Continuing training over multiple sessions
- Recovering from out-of-memory errors

Checkpoints are saved every 100 steps, with the last 3 kept to save disk space.

Training configuration:
- **Max Sequence Length**: 256 tokens
- **Micro Batch Size**: 1
- **Gradient Accumulation**: 16 steps
- **Learning Rate**: 2e-4
- **LoRA Rank**: 2 (optimized for 8GB VRAM)

### Phase 3: Evaluation & Inference

```bash
# Run the complete evaluation pipeline (Generation + Metrics)
python -m pipelines.evaluation_pipeline --test-size 10

# Run specific parts
python -m pipelines.evaluation_pipeline --base-only
python -m pipelines.evaluation_pipeline --finetuned-only
python -m pipelines.evaluation_pipeline --metrics-only
```

The evaluation pipeline runs each step (base generation, finetuned generation, metric computation) as a separate subprocess to ensure efficient memory management on Intel Arc GPUs.

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

### Web Interface (Recommended)

The easiest way to use UnArxiv is via the web interface, which provides real-time streaming simplifications.

```bash
# Start the API server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Then open your browser to: **http://localhost:8000**

### Quick Inference (Python)

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
python -m steps.inference

# Run the complete evaluation pipeline
python -m pipelines.evaluation_pipeline --test-size 10

# Compare base vs finetuned model (quick test)
python -m steps.test_models
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
python -m steps.test_models
```

---
