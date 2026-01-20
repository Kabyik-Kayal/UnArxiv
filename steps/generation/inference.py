"""
Inference script for the UnArxiv finetuned model.
Generates simplified explanations of arXiv abstracts.

This script demonstrates standalone inference using the shared model utilities.
For API-based inference, see api/main.py.
"""

import sys
import time
from tqdm import tqdm
from utils.logger import get_logger
from utils.custom_exception import CustomException
from utils.model_utils import (
    load_finetuned_model,
    generate_simplification,
    get_device,
)

logger = get_logger(__name__)

def load_model_with_progress():
    """
    Load the finetuned model with a progress bar.
    
    Wraps the shared load_finetuned_model with tqdm progress display.
    
    Returns:
        tuple: (model, tokenizer, device)
    """
    device = get_device()
    
    steps = ["Loading base model", "Moving to XPU", "Merging LoRA", "Loading tokenizer"]
    pbar = tqdm(steps, desc="Loading model", unit="step")
    
    # Load model (progress bar is for visual feedback only)
    pbar.set_description("Loading model")
    model, tokenizer, device = load_finetuned_model(device=device, merge_weights=True)
    
    # Update progress to complete
    for _ in range(4):
        pbar.update(1)
    pbar.close()
    
    return model, tokenizer, device


def main():
    """
    Main function for testing inference.
    
    Loads the model and runs inference on a sample abstract,
    displaying the result and timing information.
    """
    try:
        model, tokenizer, device = load_model_with_progress()
        
        logger.info("Starting inference test...")
        
        # Sample abstract for testing
        test_abstract = """The relationship between computing systems and the brain has served as motivation for pioneering theoreticians since John von Neumann and Alan Turing. Uniform, scale-free biological networks, such as the brain, have powerful properties, including generalizing over time, which is the main barrier for Machine Learning on the path to Universal Reasoning Models. We introduce `Dragon Hatchling' (BDH), a new Large Language Model architecture based on a scale-free biologically inspired network of $n$ locally-interacting neuron particles. BDH couples strong theoretical foundations and inherent interpretability without sacrificing Transformer-like performance. BDH is a practical, performant state-of-the-art attention-based state space sequence learning architecture. In addition to being a graph model, BDH admits a GPU-friendly formulation. It exhibits Transformer-like scaling laws: empirically BDH rivals GPT2 performance on language and translation tasks, at the same number of parameters (10M to 1B), for the same training data. BDH can be represented as a brain model. The working memory of BDH during inference entirely relies on synaptic plasticity with Hebbian learning using spiking neurons. We confirm empirically that specific, individual synapses strengthen connection whenever BDH hears or reasons about a specific concept while processing language inputs. The neuron interaction network of BDH is a graph of high modularity with heavy-tailed degree distribution. The BDH model is biologically plausible, explaining one possible mechanism which human neurons could use to achieve speech. BDH is designed for interpretability. Activation vectors of BDH are sparse and positive. We demonstrate monosemanticity in BDH on language tasks. Interpretability of state, which goes beyond interpretability of neurons and model parameters, is an inherent feature of the BDH architecture."""
        
        # Run inference with timing
        start_time = time.time()
        result = generate_simplification(model, tokenizer, device, test_abstract)
        end_time = time.time()
        
        # Display results
        print(f"Total inference time: {end_time - start_time:.4f} seconds")
        print("\n" + "="*80)
        print("SIMPLIFIED EXPLANATION:")
        print("="*80)
        print(result)
        print("="*80 + "\n")
        
        logger.info("Inference test completed successfully")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
