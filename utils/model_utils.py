"""
Shared model utilities for UnArxiv.

This module provides reusable functions for:
- Model loading and initialization
- Text generation and simplification
- Device management (XPU/CPU)
- Memory cleanup

Used by: steps/inference.py, api/main.py, steps/generate_*.py
"""

import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

# Model paths
BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "model/qwen-arxiv-simplified-arc"
MERGED_MODEL_PATH = "model/qwen-arxiv-simplified-merged"

# Generation defaults
DEFAULT_MAX_NEW_TOKENS = 512

# System prompt for simplification
SIMPLIFY_PROMPT = (
    "Simplify the following scientific abstract into plain language "
    "that anyone can understand. Use simple words, short sentences, "
    "and everyday analogies.\n\n"
)


# ============================================================================
# DEVICE UTILITIES
# ============================================================================

def get_device() -> str:
    """
    Get the best available device for inference.
    
    Returns:
        str: 'xpu' if Intel GPU available, else 'cpu'
    """
    device = "xpu" if torch.xpu.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    if device == "cpu":
        logger.warning("XPU not available, falling back to CPU. This will be slow.")
    
    return device


def cleanup_memory(device: str = None):
    """
    Clean up GPU/CPU memory after inference.
    
    Args:
        device: The device being used ('xpu' or 'cpu')
    """
    gc.collect()
    if device == "xpu" or (device is None and torch.xpu.is_available()):
        torch.xpu.synchronize()
        torch.xpu.empty_cache()


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_base_model(device: str = None):
    """
    Load the base Qwen model with optimizations.
    
    Args:
        device: Target device ('xpu' or 'cpu'). Auto-detected if None.
        
    Returns:
        tuple: (model, tokenizer, device)
    """
    if device is None:
        device = get_device()
    
    logger.info(f"Loading base model: {BASE_MODEL_ID}")
    
    # Load with SDPA (Scaled Dot Product Attention) for PyTorch 2.0+
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="cpu",  # Load to CPU first
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    
    # Move to target device
    logger.info(f"Moving model to {device}...")
    model = model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True
    )
    
    logger.info("Base model loaded successfully!")
    return model, tokenizer, device


def load_finetuned_model(device: str = None):
    """
    Load the finetuned merged model.
    
    This loads the pre-merged model from MERGED_MODEL_PATH, which is faster
    than loading base model + adapters separately.
    
    Args:
        device: Target device ('xpu' or 'cpu'). Auto-detected if None.
        
    Returns:
        tuple: (model, tokenizer, device)
    """
    if device is None:
        device = get_device()
    
    logger.info(f"Loading finetuned model: {MERGED_MODEL_PATH}")
    
    # Load merged model
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_PATH,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    
    # Move to target device
    logger.info(f"Moving model to {device}...")
    model = model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MERGED_MODEL_PATH,
        trust_remote_code=True
    )
    
    logger.info("Finetuned model loaded successfully!")
    return model, tokenizer, device


# ============================================================================
# TEXT GENERATION
# ============================================================================

def build_simplify_prompt(abstract: str) -> list:
    """
    Build the chat messages for simplification.
    
    Args:
        abstract: The scientific abstract to simplify
        
    Returns:
        list: Messages in chat format
    """
    return [
        {
            "role": "user",
            "content": f"{SIMPLIFY_PROMPT}{abstract}"
        }
    ]


def prepare_inputs(tokenizer, messages: list, device: str):
    """
    Prepare tokenized inputs for the model.
    
    Args:
        tokenizer: The tokenizer to use
        messages: Chat messages list
        device: Target device for tensors
        
    Returns:
        dict: Tokenized inputs ready for model
    """
    # Build text with template, then tokenize separately (HF recommended pattern)
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    return inputs


def generate_simplification(
    model,
    tokenizer,
    device: str,
    abstract: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    """
    Generate a simplified explanation of an arXiv abstract.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        device: Device the model is on
        abstract: Scientific abstract to simplify
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        str: Simplified explanation
    """
    messages = build_simplify_prompt(abstract)
    inputs = prepare_inputs(tokenizer, messages, device)
    
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Strip the prompt tokens to get only the generated response
    new_token_ids = generated_ids[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    
    # Cleanup
    del inputs, generated_ids
    cleanup_memory(device)
    
    return result.strip()


# ============================================================================
# ADAPTER UTILITIES
# ============================================================================

def delete_adapters():
    """
    Delete the adapter directory to save disk space.
    
    Call this after merging adapters into the base model, as the adapter
    files are no longer needed once merged.
    """
    import shutil
    from utils.logger import get_logger
    logger = get_logger(__name__)
    
    if os.path.exists(ADAPTER_PATH):
        logger.info(f"Deleting adapter directory: {ADAPTER_PATH}")
        shutil.rmtree(ADAPTER_PATH)
        logger.info("Adapter directory deleted successfully")
    else:
        logger.warning(f"Adapter directory not found: {ADAPTER_PATH}")
