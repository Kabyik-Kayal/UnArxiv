import torch
import os
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from peft import LoraConfig
from ipex_llm.transformers.qlora import get_peft_model, prepare_model_for_kbit_training
from ipex_llm.transformers import AutoModelForCausalLM
import gc

# Try to patch xe_linear requirement
try:
    import xe_linear
except ImportError:
    # Create a dummy xe_linear module to bypass the import
    import sys
    from types import ModuleType
    xe_linear = ModuleType('xe_linear')
    sys.modules['xe_linear'] = xe_linear
    print("Warning: xe_linear not found, using fallback mode")

# Disable XMX operations (which includes xe_linear)
os.environ['BIGDL_LLM_XMX_DISABLED'] = '1'
os.environ['SYCL_USE_XMX'] = '0'
os.environ['SYCL_CACHE_PERSISTENT'] = '1'

# CONFIGURATION
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATA_FILE = "data/training_data.json" 
OUTPUT_DIR = "model/qwen-arxiv-simplified-arc"

# HARDWARE SETTINGS (STRICT 8GB MODE)
# Optimization: With proper settings, we can likely increase Seq Length to 512
MAX_SEQ_LENGTH = 512       
MICRO_BATCH_SIZE = 1       
GRADIENT_ACCUMULATION = 8  
LEARNING_RATE = 2e-4

def main():
    print(f"Initializing fine-tuning for {MODEL_NAME} on Intel Arc...")

    # --- CRITICAL FIX 2: XPU Availability Check ---
    # Ensure we are using the XPU. If not, the script will silently fall back 
    # to CPU (extremely slow) or crash.
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = "xpu"
        print(f"XPU Detected: {torch.xpu.get_device_name(0)}")
        torch.xpu.empty_cache()
    else:
        raise RuntimeError(
            "Intel XPU not detected. Please ensure 'intel_extension_for_pytorch' "
            "is installed and you have sourced the OneAPI setvars script."
        )

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # 2. Load Dataset
    print("Loading dataset...")
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    print(f"Found {len(dataset)} training examples.")

    # 2a. Preprocess dataset - convert to tokenized format
    def preprocess_function(examples):
        # Combine instruction, input, and output into chat format
        texts = []
        for i in range(len(examples['instruction'])):
            text = f"<|im_start|>user\n{examples['instruction'][i]}\n\n{examples['input'][i]}<|im_end|>\n<|im_start|>assistant\n{examples['output'][i]}<|im_end|>"
            texts.append(text)
        
        # Tokenize
        model_inputs = tokenizer(
            texts,
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            padding=False,  # DataCollator will handle padding
        )
        
        # For causal LM, labels are the same as input_ids
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    # 3. Load Model (8-bit Optimized for Arc)
    print("Loading model in 8-bit quantization (low memory mode)...")
    # Clear cache multiple times to ensure maximum available memory
    # Clear both XPU and CPU memory cache
    torch.xpu.empty_cache()
    torch.xpu.synchronize()
    gc.collect()
    torch.xpu.empty_cache()
    torch.xpu.synchronize()
    
    # Use IPEX-LLM specific load_in_low_bit for Intel Arc
    # Using sym_int8 for 8-bit quantization (better quality than nf4)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_low_bit="sym_int8",  # 8-bit symmetric quantization
        optimize_model=False,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,  # Reduce CPU RAM during loading
        modules_to_not_convert=["lm_head"],
    )
    # Print model size before moving to XPU
    def get_model_size_mb(model):
        """Calculate model size in MB"""
        total_params = sum(p.numel() for p in model.parameters())
        total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        return total_bytes / (1024 ** 2)

    model_size_mb = get_model_size_mb(model)
    print(f"Model size before XPU transfer: {model_size_mb:.2f} MB")

    # CRITICAL: Disable gradient checkpointing BEFORE moving to XPU
    model.config.use_cache = False  # Required for training
    model.config.gradient_checkpointing = False  # Disable in config
    model.gradient_checkpointing = False  # Disable at model level
    if hasattr(model, '_gradient_checkpointing_func'):
        model._gradient_checkpointing_func = None
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    
    # Now move to XPU (model is already in 4-bit format)
    print("Moving model to XPU...")
    torch.xpu.empty_cache()
    model = model.to(device)
    print(f"Model loaded on device: {next(model.parameters()).device}")
    
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    # 4. LoRA Configuration (Memory Optimized)
    print("Applying LoRA adapters...")
    # Reduced r from 8 to 4 to decrease memory usage
    peft_config = LoraConfig(
        r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Clear cache to ensure max VRAM availability before training loop
    torch.xpu.empty_cache()

    # 5. Training Arguments (Memory Optimized)
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_steps=10,
        max_steps=150,
        learning_rate=LEARNING_RATE,
        bf16=True,  # bf16 is more stable in training
        logging_steps=15,
        optim="adamw_torch",  # Standard PyTorch AdamW optimizer
        save_steps=75,
        gradient_checkpointing=False,  # CRITICAL: Must be False to avoid xe_linear
        gradient_checkpointing_kwargs=None,
        max_grad_norm=0.3,  # Gradient clipping for stability
        use_cpu=False,  # Explicitly use XPU
    )

    # 6. Trainer
    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    # 7. Start Training
    print("Starting training on Intel XPU...")
    result = trainer.train()
    print(result)

    # 8. Save Model
    print("Saving adapter...")
    # Note: Saving usually saves only the adapter weights
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Done! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()