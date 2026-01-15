"""Quick test to compare base model vs finetuned model outputs."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

TEST_ABSTRACT = """we comment on zero- and low - temperature structural phase transitions , expecting that these comments might be relevant not only for this structural case . \n we first consider a textbook model whose classical version is the only model for which the landau theory of phase transitions and the concept of `` soft mode '' introduced by ginzburg are exact . within this model \n , we reveal the effects of quantum fluctuations and thermal ones at low temperatures . \n to do so , the knowledge of the dynamics of the model is needed . however , as already was emphasized by ginzburg _ \n et al . \n _ in eighties , a realistic theory for such a dynamics at high temperatures is lacking , what also seems to be the case in the low temperature regime . \n consequently , some theoretical conclusions turn out to be dependent on the assumptions on this dynamics . \n we illustrate this point with the low - temperature phase diagram , and discuss some unexpected shortcomings of the continuous medium approaches."""

PROMPT = f"<|im_start|>user\nSimplify the following scientific abstract into plain language that anyone can understand. Use simple words, short sentences, and everyday analogies.\n\n{TEST_ABSTRACT}<|im_end|>\n<|im_start|>assistant\n"

def test_model(model, tokenizer, device, name):
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.5,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if "<|im_start|>assistant" in result:
        result = result.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in result:
            result = result.split("<|im_end|>")[0]
    
    print(f"Output: {result.strip()}")

def main():
    device = "xpu" if torch.xpu.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load base model
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Get tokenizer from base model (not finetuned)
    base_tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        trust_remote_code=True
    )
    
    # Test base model
    base_model_xpu = base_model.to(device)
    base_model_xpu.eval()
    test_model(base_model_xpu, base_tokenizer, device, "BASE MODEL (Qwen2.5-3B-Instruct)")
    
    # Load finetuned adapter
    print("\nLoading finetuned LoRA adapter...")
    finetuned_model = PeftModel.from_pretrained(base_model_xpu, "model/qwen-arxiv-simplified-arc")
    finetuned_model.eval()
    
    # Get tokenizer from finetuned
    finetuned_tokenizer = AutoTokenizer.from_pretrained(
        "model/qwen-arxiv-simplified-arc",
        trust_remote_code=True
    )
    
    test_model(finetuned_model, finetuned_tokenizer, device, "FINETUNED MODEL (with LoRA)")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
