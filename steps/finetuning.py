import torch
import os
import warnings
import gc
import sys
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from utils.logger import logging
from utils.custom_exception import CustomException

warnings.filterwarnings('ignore', message='.*doesn\'t support querying the available free memory.*')

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATA_FILE = "data/training_data.json" 
OUTPUT_DIR = "model/qwen-arxiv-simplified-arc"

# Aggressive memory settings for 8GB
MAX_SEQ_LENGTH = 256        # Reduced from 512
MICRO_BATCH_SIZE = 1       
GRADIENT_ACCUMULATION = 16  # Increased from 8
LEARNING_RATE = 2e-4


def clear_xpu_memory():
    """Clear XPU memory cache to free up VRAM."""
    try:
        if torch.xpu.is_available():
            torch.xpu.empty_cache()
            torch.xpu.synchronize()
            gc.collect()
            torch.xpu.empty_cache()
            logging.info("XPU memory cleared successfully")
    except Exception as e:
        logging.warning(f"Failed to clear XPU memory: {str(e)}")


def main():
    try:
        logging.info(f"Initializing fine-tuning for {MODEL_NAME} on Intel Arc...")

        # Check XPU availability
        if not torch.xpu.is_available():
            raise RuntimeError("Intel XPU not detected. Please ensure PyTorch XPU version is installed and Intel GPU Drivers are up to date.")
        
        device = "xpu"
        logging.info(f"XPU Device: {torch.xpu.get_device_name(0)}")
        clear_xpu_memory()

        # Load tokenizer
        logging.info("Loading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
            tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
            tokenizer.padding_side = "right"
            logging.info("Tokenizer loaded successfully")
        except Exception as e:
            raise CustomException(f"Failed to load tokenizer: {str(e)}", sys)

        # Load dataset
        logging.info("Loading dataset...")
        try:
            if not os.path.exists(DATA_FILE):
                raise FileNotFoundError(f"Training data file not found: {DATA_FILE}")
            
            dataset = load_dataset("json", data_files=DATA_FILE, split="train")
            logging.info(f"Found {len(dataset)} training examples")
            
            if len(dataset) == 0:
                raise ValueError("Dataset is empty. Please check the training data file.")
        except Exception as e:
            raise CustomException(f"Failed to load dataset: {str(e)}", sys)

        def preprocess_function(examples):
            try:
                texts = []
                for i in range(len(examples['instruction'])):
                    text = f"<|im_start|>user\n{examples['instruction'][i]}\n\n{examples['input'][i]}<|im_end|>\n<|im_start|>assistant\n{examples['output'][i]}<|im_end|>"
                    texts.append(text)
                
                model_inputs = tokenizer(
                    texts,
                    max_length=MAX_SEQ_LENGTH,
                    truncation=True,
                    padding=False,
                )
                
                model_inputs["labels"] = model_inputs["input_ids"].copy()
                return model_inputs
            except KeyError as e:
                raise CustomException(f"Missing required column in dataset: {str(e)}", sys)
        
        logging.info("Tokenizing dataset...")
        try:
            tokenized_dataset = dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=dataset.column_names,
                desc="Tokenizing"
            )
            logging.info("Dataset tokenization complete")
        except Exception as e:
            raise CustomException(f"Failed to tokenize dataset: {str(e)}", sys)

        # Load model in FP32 first (on CPU to save VRAM)
        logging.info("Loading model on CPU first...")
        clear_xpu_memory()
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                dtype=torch.float32,  # Load in FP32 on CPU
                low_cpu_mem_usage=True,
                device_map="cpu",  # Keep on CPU initially
            )
            logging.info("Model loaded on CPU successfully")
        except Exception as e:
            raise CustomException(f"Failed to load model: {str(e)}", sys)
        
        # Convert to BF16 and move to XPU layer by layer
        logging.info("Converting to BF16 and moving to XPU...")
        try:
            model = model.to(torch.bfloat16)
            model = model.to(device)
            logging.info(f"Model loaded on device: {next(model.parameters()).device}")
        except Exception as e:
            raise CustomException(f"Failed to move model to XPU: {str(e)}", sys)
    
        # Configure for training
        logging.info("Configuring model for training...")
        try:
            model.config.use_cache = False
            model.gradient_checkpointing_enable()
            
            # Enable input gradients
            if hasattr(model, 'enable_input_require_grads'):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            logging.info("Model configuration complete")
        except Exception as e:
            raise CustomException(f"Failed to configure model: {str(e)}", sys)

        # LoRA with minimal rank to save memory
        logging.info("Applying LoRA adapters...")
        try:
            peft_config = LoraConfig(
                r=2,  # Minimal rank for 8GB VRAM
                lora_alpha=8,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj"]  # Only Q and V projections
            )
            
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            logging.info("LoRA adapters applied successfully")
        except Exception as e:
            raise CustomException(f"Failed to apply LoRA adapters: {str(e)}", sys)
        
        clear_xpu_memory()

        # Training args
        logging.info("Setting up training arguments...")
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            args = TrainingArguments(
                output_dir=OUTPUT_DIR,
                per_device_train_batch_size=MICRO_BATCH_SIZE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION,
                warmup_steps=10,
                max_steps=150,
                learning_rate=LEARNING_RATE,
                bf16=True,
                logging_steps=15,
                optim="adamw_torch",
                save_steps=75,
                gradient_checkpointing=True,
                max_grad_norm=0.3,
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                dataloader_num_workers=0,
                fp16=False,
                report_to="none",  # Disable wandb/tensorboard to save memory
            )
            logging.info("Training arguments configured")
        except Exception as e:
            raise CustomException(f"Failed to configure training arguments: {str(e)}", sys)

        logging.info("Initializing trainer...")
        try:
            trainer = transformers.Trainer(
                model=model,
                train_dataset=tokenized_dataset,
                args=args,
                data_collator=transformers.DataCollatorForSeq2Seq(
                    tokenizer, 
                    pad_to_multiple_of=8, 
                    return_tensors="pt", 
                    padding=True
                ),
            )
            logging.info("Trainer initialized successfully")
        except Exception as e:
            raise CustomException(f"Failed to initialize trainer: {str(e)}", sys)

        logging.info("Starting training...")
        try:
            result = trainer.train()
            logging.info(f"Training completed: {result}")
        except torch.OutOfMemoryError as e:
            logging.error("Out of memory error during training")
            logging.error("Suggestions to reduce memory usage:")
            logging.error("1. Use Qwen2.5-1.5B-Instruct instead")
            logging.error("2. Reduce MAX_SEQ_LENGTH to 128")
            logging.error("3. Set LoRA r=1")
            raise CustomException(f"Out of memory during training: {str(e)}", sys)
        except KeyboardInterrupt:
            logging.warning("Training interrupted by user")
            raise
        except Exception as e:
            raise CustomException(f"Training failed: {str(e)}", sys)

        logging.info("Saving adapter and tokenizer...")
        try:
            trainer.model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            logging.info(f"Model and tokenizer saved successfully to {OUTPUT_DIR}")
        except Exception as e:
            raise CustomException(f"Failed to save model: {str(e)}", sys)
            
    except CustomException:
        raise
    except Exception as e:
        raise CustomException(f"Unexpected error during fine-tuning: {str(e)}", sys)


if __name__ == "__main__":
    try:
        main()
    except CustomException as e:
        logging.error(f"Fine-tuning failed: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        logging.warning("Fine-tuning interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        sys.exit(1)
