"""
Generate simplified outputs from FINETUNED model.
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
from peft import PeftModel

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
    
    # Build text with template, then tokenize separately
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Strip the prompt tokens properly
    new_token_ids = generated_ids[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    
    # Aggressive cleanup
    del inputs, generated_ids
    torch.xpu.synchronize()
    gc.collect()
    
    return result.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate finetuned model outputs")
    parser.add_argument("--test-size", type=int, default=25)
    parser.add_argument("--data-path", type=str, default="data/training_data.json")
    parser.add_argument("--adapter-path", type=str, default="model/qwen-arxiv-simplified-arc")
    parser.add_argument("--output-path", type=str, default="logs/finetuned_outputs.json")
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
        
        # Load finetuned model
        logger.info("Loading FINETUNED model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        base_model = base_model.to(device)
        base_model.eval()
        
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
        model.eval()
        logger.info(f"Finetuned model loaded on {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            args.adapter_path,
            trust_remote_code=True
        )
        
        # Generate outputs
        outputs = []
        for i, abstract in enumerate(tqdm(abstracts, desc="Finetuned model", unit="sample")):
            output = generate_single(model, tokenizer, abstract, device)
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
        del model
        del base_model
        del tokenizer
        gc.collect()
        torch.xpu.empty_cache()
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
