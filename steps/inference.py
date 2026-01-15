"""
Inference script for the UnArxiv finetuned model.
Generates simplified explanations of arXiv abstracts.

NOTE: Model is loaded inside main() to avoid module-level initialization issues.
"""

import torch
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
    
    # Load base model
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Move to device
    base_model = base_model.to(device)
    base_model.eval()
    
    # Load LoRA adapter
    logger.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, "model/qwen-arxiv-simplified-arc")
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "model/qwen-arxiv-simplified-arc",
        trust_remote_code=True
    )
    
    return model, tokenizer, device


def simplify_arxiv(model, tokenizer, device, abstract):
    """Generate a simplified explanation of an arXiv abstract."""
    prompt = f"<|im_start|>user\nSimplify the following scientific abstract into plain language that anyone can understand. Use simple words, short sentences, and everyday analogies.\n\n{abstract}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.5,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and extract assistant's response
    result = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if "<|im_start|>assistant" in result:
        result = result.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in result:
            result = result.split("<|im_end|>")[0]
    
    return result.strip()


def main():
    """Main function - matches test_models.py structure."""
    try:
        model, tokenizer, device = load_model()
        
        logger.info("Starting inference test...")
        test_abstract = """we comment on zero- and low - temperature structural phase transitions , expecting that these comments might be relevant not only for this structural case . 
 we first consider a textbook model whose classical version is the only model for which the landau theory of phase transitions and the concept of `` soft mode '' introduced by ginzburg are exact . within this model 
 , we reveal the effects of quantum fluctuations and thermal ones at low temperatures . 
 to do so , the knowledge of the dynamics of the model is needed . however , as already was emphasized by ginzburg _ 
 et al . 
 _ in eighties , a realistic theory for such a dynamics at high temperatures is lacking , what also seems to be the case in the low temperature regime . 
 consequently , some theoretical conclusions turn out to be dependent on the assumptions on this dynamics . 
 we illustrate this point with the low - temperature phase diagram , and discuss some unexpected shortcomings of the continuous medium approaches."""
        
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