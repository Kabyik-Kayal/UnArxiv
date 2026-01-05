from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import sys
from utils.logger import get_logger
from utils.custom_exception import CustomException

# Initialize logger
logger = get_logger(__name__)

# Clear any cached XPU memory
try:
    if torch.xpu.is_available():
        torch.xpu.empty_cache()
        logger.info("XPU cache cleared successfully")
    else:
        logger.warning("XPU not available, falling back to CPU")
except Exception as e:
    logger.error(f"Error clearing XPU cache: {str(e)}")

# Load base model to CPU first to avoid XPU memory allocation issues
try:
    logger.info("Loading base model to CPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        device_map="cpu",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    logger.info("Base model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load base model: {str(e)}")
    raise CustomException(e, sys)

# Load your fine-tuned LoRA adapter (still on CPU)
try:
    logger.info("Loading fine-tuned LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model, 
        "model/qwen-arxiv-simplified-arc"
    )
    logger.info("LoRA adapter loaded successfully")
except Exception as e:
    logger.error(f"Failed to load LoRA adapter: {str(e)}")
    raise CustomException(e, sys)

# Now move the complete model to XPU
try:
    logger.info("Moving model to XPU...")
    model = model.to("xpu")
    model.eval()
    logger.info("Model successfully moved to XPU and set to eval mode")
except Exception as e:
    logger.error(f"Failed to move model to XPU: {str(e)}")
    raise CustomException(e, sys)

try:
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "model/qwen-arxiv-simplified-arc",
        trust_remote_code=True
    )
    logger.info("Tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load tokenizer: {str(e)}")
    raise CustomException(e, sys)

# Generate simplified explanations
def simplify_arxiv(abstract):
    """
    Generate a simplified explanation of an arXiv abstract.
    
    Args:
        abstract (str): The arXiv abstract to simplify
        
    Returns:
        str: Simplified explanation
        
    Raises:
        CustomException: If text generation fails
    """
    try:
        logger.info(f"Simplify the following scientific abstract into plain language that anyone can understand. Use simple words, short sentences, and everyday analogies. (length: {len(abstract)} chars)")
        prompt = f"<|im_start|>user\n{abstract}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("xpu")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("Successfully generated simplified explanation")
        return result
    except Exception as e:
        logger.error(f"Failed to generate simplified explanation: {str(e)}")
        raise CustomException(e, sys)

# Test it
if __name__ == "__main__":
    try:
        logger.info("Starting inference test...")
        test_abstract = "we comment on zero- and low - temperature structural phase transitions , expecting that these comments might be relevant not only for this structural case . \n we first consider a textbook model whose classical version is the only model for which the landau theory of phase transitions and the concept of `` soft mode '' introduced by ginzburg are exact . within this model \n , we reveal the effects of quantum fluctuations and thermal ones at low temperatures . \n to do so , the knowledge of the dynamics of the model is needed . however , as already was emphasized by ginzburg _ \n et al . \n _ in eighties , a realistic theory for such a dynamics at high temperatures is lacking , what also seems to be the case in the low temperature regime . \n consequently , some theoretical conclusions turn out to be dependent on the assumptions on this dynamics . \n we illustrate this point with the low - temperature phase diagram , and discuss some unexpected shortcomings of the continuous medium approaches."
        
        result = simplify_arxiv(test_abstract)
        print("\n" + "="*80)
        print("SIMPLIFIED EXPLANATION:")
        print("="*80)
        print(result)
        print("="*80 + "\n")
        logger.info("Inference test completed successfully")
    except CustomException as ce:
        logger.error(f"Custom exception occurred: {ce.error_message}")
        print(f"\nError: {ce.error_message}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}")
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)