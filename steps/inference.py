"""
Inference script for the UnArxiv finetuned model.
Generates simplified explanations of arXiv abstracts.

NOTE: Model is loaded inside main() to avoid module-level initialization issues.
"""

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
from utils.logger import get_logger
from utils.custom_exception import CustomException

# Initialize logger
logger = get_logger(__name__)


def load_model():
    """Load the finetuned model and tokenizer."""
    device = "xpu" if torch.xpu.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Clear XPU cache
    if device == "xpu":
        try:
            torch.xpu.empty_cache()
            logger.info("XPU cache cleared")
        except:
            pass
    
    # Loading steps with progress
    steps = ["Loading base model", "Moving to device", "Loading LoRA adapter", "Loading tokenizer"]
    pbar = tqdm(steps, desc="Loading model", unit="step")
    
    # Step 1: Load base model
    pbar.set_description("Loading base model")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    pbar.update(1)
    
    # Step 2: Move to device
    pbar.set_description("Moving to device")
    base_model = base_model.to(device)
    base_model.eval()
    pbar.update(1)
    
    # Step 3: Load LoRA adapter
    pbar.set_description("Loading LoRA adapter")
    model = PeftModel.from_pretrained(base_model, "model/qwen-arxiv-simplified-arc")
    model.eval()
    pbar.update(1)
    
    # Step 4: Load tokenizer
    pbar.set_description("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        "model/qwen-arxiv-simplified-arc",
        trust_remote_code=True
    )
    pbar.update(1)
    pbar.close()
    
    return model, tokenizer, device


def simplify_arxiv(model, tokenizer, device, abstract):
    """Generate a simplified explanation of an arXiv abstract."""
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
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated portion (exclude input prompt)
    result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return result.strip()


def main():
    """Main function - matches test_models.py structure."""
    try:
        model, tokenizer, device = load_model()
        
        logger.info("Starting inference test...")
        test_abstract = """The relationship between computing systems and the brain has served as motivation for pioneering theoreticians since John von Neumann and Alan Turing. Uniform, scale-free biological networks, such as the brain, have powerful properties, including generalizing over time, which is the main barrier for Machine Learning on the path to Universal Reasoning Models. We introduce `Dragon Hatchling' (BDH), a new Large Language Model architecture based on a scale-free biologically inspired network of $n$ locally-interacting neuron particles. BDH couples strong theoretical foundations and inherent interpretability without sacrificing Transformer-like performance. BDH is a practical, performant state-of-the-art attention-based state space sequence learning architecture. In addition to being a graph model, BDH admits a GPU-friendly formulation. It exhibits Transformer-like scaling laws: empirically BDH rivals GPT2 performance on language and translation tasks, at the same number of parameters (10M to 1B), for the same training data. BDH can be represented as a brain model. The working memory of BDH during inference entirely relies on synaptic plasticity with Hebbian learning using spiking neurons. We confirm empirically that specific, individual synapses strengthen connection whenever BDH hears or reasons about a specific concept while processing language inputs. The neuron interaction network of BDH is a graph of high modularity with heavy-tailed degree distribution. The BDH model is biologically plausible, explaining one possible mechanism which human neurons could use to achieve speech. BDH is designed for interpretability. Activation vectors of BDH are sparse and positive. We demonstrate monosemanticity in BDH on language tasks. Interpretability of state, which goes beyond interpretability of neurons and model parameters, is an inherent feature of the BDH architecture."""
        result = simplify_arxiv(model, tokenizer, device, test_abstract)
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