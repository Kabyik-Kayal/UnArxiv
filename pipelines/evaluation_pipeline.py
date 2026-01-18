"""
Unified Evaluation Pipeline for UnArxiv.
Orchestrates base output generation, finetuned output generation, and metric computation.
Each step runs as a completely separate Python process to ensure memory isolation.
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
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"{'='*60}\n")
    
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
    parser = argparse.ArgumentParser(description="UnArxiv Evaluation Pipeline")
    parser.add_argument("--test-size", type=int, default=25)
    parser.add_argument("--data-path", type=str, default="data/training_data.json")
    parser.add_argument("--adapter-path", type=str, default="model/qwen-arxiv-simplified-arc")
    parser.add_argument("--base-only", action="store_true", help="Generate only base model outputs")
    parser.add_argument("--finetuned-only", action="store_true", help="Generate only finetuned model outputs")
    parser.add_argument("--metrics-only", action="store_true", help="Compute metrics only (skip generation)")
    args = parser.parse_args()
    
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_exe = sys.executable
    
    # Output paths
    base_outputs_path = "logs/base_outputs.json"
    finetuned_outputs_path = "logs/finetuned_outputs.json"
    results_path = "logs/evaluation_results.json"
    
    steps_completed = 0
    steps_total = 3 if not (args.base_only or args.finetuned_only or args.metrics_only) else 1
    
    print("\n" + "="*60)
    print("UNARXIV EVALUATION PIPELINE")
    print("="*60)
    print(f"Test size: {args.test_size}")
    print(f"Data path: {args.data_path}")
    print(f"Adapter path: {args.adapter_path}")
    print("="*60 + "\n")
    
    # Step 1: Generate base model outputs
    if not args.finetuned_only and not args.metrics_only:
        success = run_step(
            "Generate Base Model Outputs",
            [
                python_exe, "-m", "steps.generate_base_outputs",
                "--test-size", str(args.test_size),
                "--data-path", args.data_path,
                "--output-path", base_outputs_path,
            ],
            project_root
        )
        if not success:
            print("\n❌ Pipeline failed at: Generate Base Model Outputs")
            sys.exit(1)
        steps_completed += 1
        
        if args.base_only:
            print(f"\n✅ Base outputs saved to {base_outputs_path}")
            return
    
    # Step 2: Generate finetuned model outputs
    if not args.base_only and not args.metrics_only:
        success = run_step(
            "Generate Finetuned Model Outputs",
            [
                python_exe, "-m", "steps.generate_finetuned_outputs",
                "--test-size", str(args.test_size),
                "--data-path", args.data_path,
                "--adapter-path", args.adapter_path,
                "--output-path", finetuned_outputs_path,
            ],
            project_root
        )
        if not success:
            print("\n❌ Pipeline failed at: Generate Finetuned Model Outputs")
            sys.exit(1)
        steps_completed += 1
        
        if args.finetuned_only:
            print(f"\n✅ Finetuned outputs saved to {finetuned_outputs_path}")
            return
    
    # Step 3: Compute metrics
    if not args.base_only and not args.finetuned_only:
        success = run_step(
            "Compute Evaluation Metrics",
            [
                python_exe, "-m", "steps.compute_metrics",
                "--base-outputs", base_outputs_path,
                "--finetuned-outputs", finetuned_outputs_path,
                "--output-path", results_path,
            ],
            project_root
        )
        if not success:
            print("\n❌ Pipeline failed at: Compute Evaluation Metrics")
            sys.exit(1)
        steps_completed += 1
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Base outputs: {base_outputs_path}")
    print(f"Finetuned outputs: {finetuned_outputs_path}")
    print(f"Evaluation results: {results_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
