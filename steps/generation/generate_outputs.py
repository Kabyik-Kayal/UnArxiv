"""
Unified output generator for UnArxiv.

Generates simplified outputs from either BASE or FINETUNED model.
Replaces the separate generate_base_outputs.py and generate_finetuned_outputs.py scripts.

Usage:
    python -m steps.generation.generate_outputs --model-type base --test-size 10
    python -m steps.generation.generate_outputs --model-type finetuned --test-size 10
"""

import json
import sys
import os
import argparse
from datetime import datetime
from tqdm import tqdm
from utils.logger import get_logger
from utils.custom_exception import CustomException
from utils.model_utils import (
    load_base_model,
    load_finetuned_model,
    generate_simplification,
    cleanup_memory,
    BASE_MODEL_ID,
    ADAPTER_PATH,
)

logger = get_logger(__name__)

# Default output paths
DEFAULT_BASE_OUTPUT = "logs/base_outputs.json"
DEFAULT_FINETUNED_OUTPUT = "logs/finetuned_outputs.json"


def load_test_data(data_path: str, test_split: float, test_size: int):
    """
    Load test data from training data JSON.
    
    Args:
        data_path: Path to training data JSON
        test_split: Fraction of data used for training (rest is test)
        test_size: Number of test samples to process
        
    Returns:
        tuple: (abstracts, references)
    """
    logger.info(f"Loading data from {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    # Split data: use samples after training split for testing
    split_idx = int(len(all_data) * test_split)
    test_data = all_data[split_idx:][:test_size]
    
    abstracts = [d["input"] for d in test_data]
    references = [d["output"] for d in test_data]
    
    logger.info(f"Loaded {len(abstracts)} test samples")
    return abstracts, references


def generate_outputs(model, tokenizer, device, abstracts: list, model_name: str):
    """
    Generate simplified outputs for a list of abstracts.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        device: Device the model is on
        abstracts: List of abstracts to simplify
        model_name: Name for progress bar display
        
    Returns:
        list: Generated outputs
    """
    outputs = []
    for i, abstract in enumerate(tqdm(abstracts, desc=model_name, unit="sample")):
        output = generate_simplification(model, tokenizer, device, abstract)
        logger.info(f"{model_name} output {i+1}: {output[:100]}...")
        outputs.append(output)
    return outputs


def save_results(
    output_path: str,
    model_name: str,
    abstracts: list,
    references: list,
    outputs: list,
):
    """Save generation results to JSON."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "test_size": len(outputs),
        "abstracts": abstracts,
        "references": references,
        "outputs": outputs,
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Outputs saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified output generator for UnArxiv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate base model outputs:
    python -m steps.generation.generate_outputs --model-type base --test-size 10
    
  Generate finetuned model outputs:
    python -m steps.generation.generate_outputs --model-type finetuned --test-size 10
"""
    )
    parser.add_argument(
        "--model-type", 
        type=str, 
        required=True,
        choices=["base", "finetuned"],
        help="Model type to use for generation"
    )
    parser.add_argument(
        "--test-size", 
        type=int, 
        default=25,
        help="Number of test samples to process"
    )
    parser.add_argument(
        "--data-path", 
        type=str, 
        default="data/training_data.json",
        help="Path to training data JSON"
    )
    parser.add_argument(
        "--adapter-path", 
        type=str, 
        default=ADAPTER_PATH,
        help="Path to LoRA adapter (finetuned model only)"
    )
    parser.add_argument(
        "--output-path", 
        type=str, 
        default=None,
        help="Path to save output JSON (auto-determined if not specified)"
    )
    parser.add_argument(
        "--test-split", 
        type=float, 
        default=0.9,
        help="Fraction of data used for training (rest is test)"
    )
    args = parser.parse_args()
    
    # Determine output path if not specified
    if args.output_path is None:
        args.output_path = (
            DEFAULT_BASE_OUTPUT if args.model_type == "base" 
            else DEFAULT_FINETUNED_OUTPUT
        )
    
    try:
        # Load test data
        abstracts, references = load_test_data(
            args.data_path, args.test_split, args.test_size
        )
        
        # Load appropriate model
        if args.model_type == "base":
            logger.info(f"Loading BASE model: {BASE_MODEL_ID}")
            model, tokenizer, device = load_base_model()
            model_name = BASE_MODEL_ID
            display_name = "Base model"
        else:
            logger.info(f"Loading FINETUNED model with adapter: {args.adapter_path}")
            model, tokenizer, device = load_finetuned_model()
            model_name = f"Finetuned ({args.adapter_path})"
            display_name = "Finetuned model"
        
        # Generate outputs
        outputs = generate_outputs(
            model, tokenizer, device, abstracts, display_name
        )
        
        # Save results
        save_results(
            args.output_path, model_name, abstracts, references, outputs
        )
        
        # Cleanup
        del model, tokenizer
        cleanup_memory(device)
        
        logger.info(f"Generation complete! Results saved to {args.output_path}")
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
