"""
Quick test to compare base model vs finetuned model outputs.

This script provides a simple way to compare the outputs of both models
on the same abstract. Uses the shared model utilities for consistency.
"""

import sys
import torch
from utils.logger import get_logger
from utils.model_utils import (
    load_base_model,
    load_finetuned_model,
    generate_simplification,
    cleanup_memory,
    get_device,
)

logger = get_logger(__name__)

TEST_ABSTRACT = """we comment on zero- and low - temperature structural phase transitions , expecting that these comments might be relevant not only for this structural case . 
we first consider a textbook model whose classical version is the only model for which the landau theory of phase transitions and the concept of `` soft mode '' introduced by ginzburg are exact . within this model 
, we reveal the effects of quantum fluctuations and thermal ones at low temperatures . 
to do so , the knowledge of the dynamics of the model is needed . however , as already was emphasized by ginzburg _ 
et al . 
_ in eighties , a realistic theory for such a dynamics at high temperatures is lacking , what also seems to be the case in the low temperature regime . 
consequently , some theoretical conclusions turn out to be dependent on the assumptions on this dynamics . 
we illustrate this point with the low - temperature phase diagram , and discuss some unexpected shortcomings of the continuous medium approaches."""


def test_model(model, tokenizer, device, name: str) -> str:
    """
    Test a model with the sample abstract.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        device: Device the model is on
        name: Display name for the model
        
    Returns:
        str: The generated output
    """
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    result = generate_simplification(model, tokenizer, device, TEST_ABSTRACT, max_new_tokens=150)
    
    print(f"Output: {result}")
    return result


def main():
    device = get_device()
    
    try:
        # Load and test base model
        print("\nLoading base model...")
        base_model, base_tokenizer, device = load_base_model(device)
        test_model(base_model, base_tokenizer, device, "BASE MODEL (Qwen2.5-3B-Instruct)")
        
        # Cleanup base model before loading finetuned
        del base_model
        cleanup_memory(device)
        
        # Load and test finetuned model
        print("\nLoading finetuned LoRA adapter...")
        ft_model, ft_tokenizer, device = load_finetuned_model(device, merge_weights=True)
        test_model(ft_model, ft_tokenizer, device, "FINETUNED MODEL (with LoRA)")
        
        # Cleanup
        del ft_model
        cleanup_memory(device)
        
        print("\n" + "="*60)
        print("Test completed successfully!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
