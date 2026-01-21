"""
Model Merger - Merge LoRA adapters with base model.

After finetuning completes, this script merges the trained LoRA adapters
with the base model to create a standalone finetuned model. This eliminates
the need to load the base model + adapters separately during inference,
resulting in faster model loading and simpler deployment.

Usage:
    python -m steps.training.model_merger

The merged model will be saved to: model/qwen-arxiv-simplified-merged
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils.logger import logging
from utils.custom_exception import CustomException
from utils.model_utils import (
    BASE_MODEL_ID,
    ADAPTER_PATH,
    MERGED_MODEL_PATH,
    get_device,
    cleanup_memory,
)


# Alias for backwards compatibility
MERGED_OUTPUT_DIR = MERGED_MODEL_PATH


def validate_adapter_path():
    """Validate that the adapter path exists and contains required files."""
    if not os.path.exists(ADAPTER_PATH):
        raise FileNotFoundError(
            f"Adapter path not found: {ADAPTER_PATH}\n"
            "Please run finetuning first: python -m steps.training.finetuning"
        )
    
    # Check for adapter_config.json (required for LoRA adapters)
    adapter_config = os.path.join(ADAPTER_PATH, "adapter_config.json")
    if not os.path.exists(adapter_config):
        raise FileNotFoundError(
            f"adapter_config.json not found in {ADAPTER_PATH}\n"
            "This directory does not appear to contain valid LoRA adapters."
        )
    
    logging.info(f"Found valid adapter at: {ADAPTER_PATH}")


def merge_model():
    """
    Merge LoRA adapters with base model and save as standalone model.
    
    This creates a complete model that:
    - Contains all finetuned weights merged in
    - Can be loaded directly without loading base + adapters
    - Has faster loading time during inference
    - Is easier to deploy and share
    """
    try:
        cleanup_memory()
        logging.info("Model Merger - Merging LoRA adapters with base model")
        # Validate adapter exists
        validate_adapter_path()
        # Get best available device
        device = get_device()
        cleanup_memory(device)
        # Step 1: Load base model
        logging.info(f"Loading base model: {BASE_MODEL_ID}")
        logging.info("This may take a moment...")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            device_map="cpu",  # Load to CPU first for memory efficiency
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        logging.info("Base model loaded successfully")
        
        # Step 2: Load tokenizer from adapter
        logging.info(f"Loading tokenizer from adapter: {ADAPTER_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(
            ADAPTER_PATH,
            trust_remote_code=True
        )
        logging.info("Tokenizer loaded successfully")
        
        # Step 3: Load LoRA adapter
        logging.info(f"Loading LoRA adapter: {ADAPTER_PATH}")
        peft_model = PeftModel.from_pretrained(
            base_model,
            ADAPTER_PATH,
            torch_dtype=torch.bfloat16,
        )
        logging.info("LoRA adapter loaded successfully")
        
        # Step 4: Merge weights
        logging.info("Merging LoRA weights into base model...")
        logging.info("This permanently combines the adapter weights with the base model")
        
        merged_model = peft_model.merge_and_unload()
        logging.info("Weights merged successfully!")
        
        # Clear intermediate objects and move model to CPU to free GPU memory
        del peft_model
        del base_model
        cleanup_memory(device)
        
        # Move merged model to CPU before saving to avoid GPU memory issues
        logging.info("Moving merged model to CPU for saving...")
        merged_model = merged_model.to("cpu")
        cleanup_memory(device)
        
        # Step 5: Save merged model
        logging.info(f"Saving merged model to: {MERGED_OUTPUT_DIR}")
        
        # Remove existing merged model directory to avoid file lock issues
        import shutil
        if os.path.exists(MERGED_OUTPUT_DIR):
            logging.info(f"Removing existing merged model directory...")
            shutil.rmtree(MERGED_OUTPUT_DIR)
        
        os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)
        
        # Save model and tokenizer
        merged_model.save_pretrained(
            MERGED_OUTPUT_DIR,
            safe_serialization=True,  # Use safetensors format
        )
        tokenizer.save_pretrained(MERGED_OUTPUT_DIR)
        
        logging.info("Model merging complete!")
        logging.info(f"Merged model saved to: {MERGED_OUTPUT_DIR}")
        
        # Cleanup
        del merged_model
        cleanup_memory(device)
        
        return MERGED_OUTPUT_DIR
        
    except FileNotFoundError as e:
        raise CustomException(str(e), sys)
    except Exception as e:
        import traceback
        logging.error(f"Error during model save: {traceback.format_exc()}")
        raise CustomException(f"Failed to merge model: {str(e)}", sys)


if __name__ == "__main__":
    try:
        output_path = merge_model()
        logging.info(f"Success! Merged model available at: {output_path}")
    except CustomException as e:
        logging.error(f"Model merging failed: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        logging.warning("Model merging interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        sys.exit(1)
