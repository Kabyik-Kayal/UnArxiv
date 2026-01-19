"""
Generate simplified outputs from FINETUNED model.

Saves results to JSON for later evaluation against the base model.
Uses the shared model utilities for consistency.
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
    load_finetuned_model,
    generate_simplification,
    cleanup_memory,
    ADAPTER_PATH,
)

logger = get_logger(__name__)


def main():
    """
    Main function to generate finetuned model outputs.
    
    Loads the finetuned model with LoRA adapter, generates simplified
    outputs for test data, and saves results to JSON for evaluation.
    """
    parser = argparse.ArgumentParser(description="Generate finetuned model outputs")
    parser.add_argument("--test-size", type=int, default=25,
                        help="Number of test samples to process")
    parser.add_argument("--data-path", type=str, default="data/training_data.json",
                        help="Path to training data JSON")
    parser.add_argument("--adapter-path", type=str, default=ADAPTER_PATH,
                        help="Path to LoRA adapter")
    parser.add_argument("--output-path", type=str, default="logs/finetuned_outputs.json",
                        help="Path to save output JSON")
    parser.add_argument("--test-split", type=float, default=0.9,
                        help="Fraction of data used for training (rest is test)")
    args = parser.parse_args()
    
    try:
        # Load test data
        logger.info(f"Loading data from {args.data_path}")
        with open(args.data_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        # Split data: use samples after training split for testing
        split_idx = int(len(all_data) * args.test_split)
        test_data = all_data[split_idx:][:args.test_size]
        
        abstracts = [d["input"] for d in test_data]
        references = [d["output"] for d in test_data]
        
        # Load finetuned model using shared utility
        # merge_weights=True for faster inference (recommended for batch generation)
        logger.info(f"Loading FINETUNED model with adapter: {args.adapter_path}")
        model, tokenizer, device = load_finetuned_model(merge_weights=True)
        
        # Generate outputs
        outputs = []
        for i, abstract in enumerate(tqdm(abstracts, desc="Finetuned model", unit="sample")):
            output = generate_simplification(model, tokenizer, device, abstract)
            logger.info(f"Finetuned output {i+1}: {output[:100]}...")
            outputs.append(output)
        
        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": f"Finetuned ({args.adapter_path})",
            "test_size": len(outputs),
            "abstracts": abstracts,
            "references": references,
            "outputs": outputs,
        }
        
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Finetuned outputs saved to {args.output_path}")
        
        # Cleanup
        del model, tokenizer
        cleanup_memory(device)
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
