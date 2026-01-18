"""
Generate simplified outputs from BASE model (Qwen2.5-3B-Instruct).
Saves results to JSON for later evaluation.
"""

import json
import sys
import os
import gc
import argparse
from datetime import datetime
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)


def get_device():
    """Always use XPU."""
    logger.info("Using device: xpu")
    return "xpu"


def generate_single(model, tokenizer, abstract: str, device: str) -> str:
    """Generate a single simplified abstract using proper chat template."""
    messages = [
        {"role": "user", "content": f"Simplify the following scientific abstract into plain language that anyone can understand. Use simple words, short sentences, and everyday analogies.\n\n{abstract}"},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated portion (exclude input prompt)
    result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    # Clean up tensors
    del inputs
    del outputs
    gc.collect()
    if device == "xpu":
        torch.xpu.empty_cache()
    
    return result.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate base model outputs")
    parser.add_argument("--test-size", type=int, default=25)
    parser.add_argument("--data-path", type=str, default="data/training_data.json")
    parser.add_argument("--output-path", type=str, default="logs/base_outputs.json")
    parser.add_argument("--test-split", type=float, default=0.9)
    args = parser.parse_args()
    
    try:
        # Load test data
        logger.info(f"Loading data from {args.data_path}")
        with open(args.data_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        split_idx = int(len(all_data) * args.test_split)
        test_data = all_data[split_idx:][:args.test_size]
        
        abstracts = [d["input"] for d in test_data]
        references = [d["output"] for d in test_data]
        
        device = get_device()
        
        # Load base model
        logger.info("Loading BASE model (Qwen2.5-3B-Instruct)...")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = model.to(device)
        model.eval()
        logger.info(f"Base model loaded on {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            trust_remote_code=True
        )
        
        # Generate outputs
        outputs = []
        for i, abstract in enumerate(tqdm(abstracts, desc="Base model", unit="sample")):
            output = generate_single(model, tokenizer, abstract, device)
            logger.info(f"Base output {i+1}: {output[:100]}...")
            outputs.append(output)
            
            # Memory cleanup every 5 samples
            if (i + 1) % 5 == 0:
                gc.collect()
                if device == "xpu":
                    torch.xpu.empty_cache()
                logger.info(f"Memory cleanup after {i+1} samples")
        
        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "test_size": len(outputs),
            "abstracts": abstracts,
            "references": references,
            "outputs": outputs,
        }
        
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Base outputs saved to {args.output_path}")
        
        # Cleanup
        del model
        del tokenizer
        gc.collect()
        torch.xpu.empty_cache()
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
