"""
Finetuning Pipeline - Unified pipeline for model finetuning, merging, and cleanup.

This pipeline orchestrates the complete finetuning workflow:
1. Finetune the base model with LoRA adapters
2. Merge adapters with base model to create standalone finetuned model
3. Delete the adapter files (optional) to save disk space

Each step runs as a completely separate Python process to ensure memory isolation.

Usage:
    python -m pipelines.finetuning_model
    python -m pipelines.finetuning_model --skip-finetuning  # Only merge existing adapters
    python -m pipelines.finetuning_model --keep-adapters    # Don't delete adapters after merge
"""

import subprocess
import sys
import argparse
import os

from utils.logger import get_logger

logger = get_logger(__name__)


def run_step(step_name: str, command: list, cwd: str) -> bool:
    """Run a pipeline step as a subprocess."""
    logger.info(f"Starting: {step_name}")
    print('='*60)
    print(f"STEP: {step_name}")
    print('='*60)
    
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            text=True,
        )
        logger.info(f"Completed: {step_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {step_name} (exit code {e.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description="UnArxiv Finetuning Pipeline")
    parser.add_argument(
        "--skip-finetuning", 
        action="store_true", 
        help="Skip finetuning and only merge existing adapters"
    )
    parser.add_argument(
        "--keep-adapters", 
        action="store_true", 
        help="Keep adapter files after merging (don't delete to save disk space)"
    )
    parser.add_argument(
        "--merge-only", 
        action="store_true", 
        help="Only run the merge step (alias for --skip-finetuning --keep-adapters)"
    )
    args = parser.parse_args()
    
    # Handle --merge-only as alias
    if args.merge_only:
        args.skip_finetuning = True
        args.keep_adapters = True
    
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_exe = sys.executable
    
    print("="*60)
    print("UNARXIV FINETUNING PIPELINE")
    print("="*60)
    print(f"Skip finetuning: {args.skip_finetuning}")
    print(f"Keep adapters: {args.keep_adapters}")
    print("="*60)
    
    steps_completed = 0
    
    # Step 1: Finetune model with LoRA adapters
    if not args.skip_finetuning:
        success = run_step(
            "Finetune Model with LoRA Adapters",
            [python_exe, "-m", "steps.training.finetuning"],
            project_root
        )
        if not success:
            print("\nPipeline failed at: Finetune Model")
            sys.exit(1)
        steps_completed += 1
    
    # Step 2: Merge adapters with base model
    success = run_step(
        "Merge Adapters with Base Model",
        [python_exe, "-m", "steps.training.model_merger"],
        project_root
    )
    if not success:
        print("\nPipeline failed at: Merge Adapters")
        sys.exit(1)
    steps_completed += 1
    
    # Step 3: Delete adapters to save disk space (optional)
    if not args.keep_adapters:
        # Import here to avoid loading model_utils at module level
        from utils.model_utils import ADAPTER_PATH
        import shutil
        
        print('='*60)
        print("STEP: Delete Adapter Files")
        print('='*60)
        
        if os.path.exists(ADAPTER_PATH):
            logger.info(f"Deleting adapter directory: {ADAPTER_PATH}")
            try:
                shutil.rmtree(ADAPTER_PATH)
                logger.info("Adapter directory deleted successfully")
                steps_completed += 1
            except Exception as e:
                logger.warning(f"Failed to delete adapters: {e}")
        else:
            logger.warning(f"Adapter directory not found: {ADAPTER_PATH}")
    
    print("="*60)
    print("FINETUNING PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Steps completed: {steps_completed}")
    print(f"Merged model saved to: model/qwen-arxiv-simplified-merged")
    print("="*60)


if __name__ == "__main__":
    main()
