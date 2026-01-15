"""
Evaluation script for the UnArxiv finetuned model.
Computes ROUGE scores and readability metrics to measure simplification quality.
"""

import json
import sys
import os
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rouge_score import rouge_scorer
import textstat

from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluates the finetuned model on abstract simplification tasks."""
    
    def __init__(self, model_path: str = "model/qwen-arxiv-simplified-arc"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = "xpu" if torch.xpu.is_available() else "cpu"
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
    def load_model(self):
        """Load the finetuned model and tokenizer."""
        try:
            logger.info("Loading base model to CPU...")
            base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-3B-Instruct",
                device_map="cpu",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            
            # Move base model to device FIRST (critical for XPU)
            logger.info(f"Moving base model to {self.device}...")
            base_model = base_model.to(self.device)
            base_model.eval()
            
            # THEN load LoRA adapter
            logger.info("Loading fine-tuned LoRA adapter...")
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model.eval()
            
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise CustomException(e, sys)
    
    def generate_simplification(self, abstract: str) -> str:
        """Generate simplified version of an abstract."""
        try:
            prompt = f"<|im_start|>user\nSimplify the following scientific abstract into plain language that anyone can understand. Use simple words, short sentences, and everyday analogies.\n\n{abstract}<|im_end|>\n<|im_start|>assistant\n"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.5,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and extract assistant's response (matching inference.py)
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            if "<|im_start|>assistant" in result:
                result = result.split("<|im_start|>assistant")[-1]
                if "<|im_end|>" in result:
                    result = result.split("<|im_end|>")[0]
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate simplification: {str(e)}")
            return ""
    
    def compute_rouge_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        """Compute ROUGE scores between prediction and reference."""
        scores = self.rouge_scorer.score(reference, prediction)
        return {
            "rouge1_f1": scores['rouge1'].fmeasure,
            "rouge2_f1": scores['rouge2'].fmeasure,
            "rougeL_f1": scores['rougeL'].fmeasure,
        }
    
    def compute_readability_metrics(self, text: str) -> Dict[str, float]:
        """Compute readability metrics for a text."""
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "gunning_fog": textstat.gunning_fog(text),
            "word_count": textstat.lexicon_count(text, removepunct=True),
            "sentence_count": textstat.sentence_count(text),
        }
    
    def evaluate(self, test_data: List[Dict], num_samples: int = None) -> Dict:
        """
        Run full evaluation on test data.
        
        Args:
            test_data: List of dicts with 'input' (original) and 'output' (reference simplified)
            num_samples: Number of samples to evaluate (None = all)
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        if num_samples:
            test_data = test_data[:num_samples]
        
        logger.info(f"Evaluating on {len(test_data)} samples...")
        
        results = {
            "rouge_scores": [],
            "original_readability": [],
            "generated_readability": [],
            "reference_readability": [],
            "individual_results": []
        }
        
        for i, sample in enumerate(test_data):
            original = sample["input"]
            reference = sample["output"]
            
            logger.info(f"Processing sample {i+1}/{len(test_data)}")
            
            # Generate prediction
            generated = self.generate_simplification(original)
            
            if not generated:
                logger.warning(f"Empty generation for sample {i+1}, skipping")
                continue
            
            # Compute ROUGE
            rouge = self.compute_rouge_scores(generated, reference)
            results["rouge_scores"].append(rouge)
            
            # Compute readability for all versions
            orig_read = self.compute_readability_metrics(original)
            gen_read = self.compute_readability_metrics(generated)
            ref_read = self.compute_readability_metrics(reference)
            
            results["original_readability"].append(orig_read)
            results["generated_readability"].append(gen_read)
            results["reference_readability"].append(ref_read)
            
            # Store individual result
            results["individual_results"].append({
                "original": original[:200] + "...",  # Truncate for readability
                "generated": generated[:200] + "...",
                "reference": reference[:200] + "...",
                "rouge": rouge,
                "readability_improvement": {
                    "flesch_ease_delta": gen_read["flesch_reading_ease"] - orig_read["flesch_reading_ease"],
                    "grade_level_delta": orig_read["flesch_kincaid_grade"] - gen_read["flesch_kincaid_grade"],
                }
            })
        
        # Compute aggregate metrics
        aggregate = self._compute_aggregates(results)
        results["aggregate"] = aggregate
        
        return results
    
    def _compute_aggregates(self, results: Dict) -> Dict:
        """Compute aggregate metrics from individual results."""
        def avg(lst, key):
            vals = [d[key] for d in lst if key in d]
            return sum(vals) / len(vals) if vals else 0
        
        rouge_scores = results["rouge_scores"]
        orig_read = results["original_readability"]
        gen_read = results["generated_readability"]
        ref_read = results["reference_readability"]
        
        return {
            "num_samples": len(rouge_scores),
            "rouge": {
                "rouge1_f1": avg(rouge_scores, "rouge1_f1"),
                "rouge2_f1": avg(rouge_scores, "rouge2_f1"),
                "rougeL_f1": avg(rouge_scores, "rougeL_f1"),
            },
            "readability": {
                "original": {
                    "avg_flesch_ease": avg(orig_read, "flesch_reading_ease"),
                    "avg_grade_level": avg(orig_read, "flesch_kincaid_grade"),
                    "avg_word_count": avg(orig_read, "word_count"),
                },
                "generated": {
                    "avg_flesch_ease": avg(gen_read, "flesch_reading_ease"),
                    "avg_grade_level": avg(gen_read, "flesch_kincaid_grade"),
                    "avg_word_count": avg(gen_read, "word_count"),
                },
                "reference": {
                    "avg_flesch_ease": avg(ref_read, "flesch_reading_ease"),
                    "avg_grade_level": avg(ref_read, "flesch_kincaid_grade"),
                    "avg_word_count": avg(ref_read, "word_count"),
                },
            },
            "improvements": {
                "flesch_ease_improvement": avg(gen_read, "flesch_reading_ease") - avg(orig_read, "flesch_reading_ease"),
                "grade_level_reduction": avg(orig_read, "flesch_kincaid_grade") - avg(gen_read, "flesch_kincaid_grade"),
                "word_count_reduction_pct": (1 - avg(gen_read, "word_count") / avg(orig_read, "word_count")) * 100 if avg(orig_read, "word_count") else 0,
            }
        }


def print_results(results: Dict):
    """Pretty print evaluation results to console."""
    agg = results["aggregate"]
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nSamples evaluated: {agg['num_samples']}")
    
    print("\n📊 ROUGE SCORES (vs reference simplifications):")
    print(f"   ROUGE-1 F1: {agg['rouge']['rouge1_f1']:.4f}")
    print(f"   ROUGE-2 F1: {agg['rouge']['rouge2_f1']:.4f}")
    print(f"   ROUGE-L F1: {agg['rouge']['rougeL_f1']:.4f}")
    
    print("\n📖 READABILITY COMPARISON:")
    print("                      Original    Generated   Reference")
    print(f"   Flesch Reading Ease: {agg['readability']['original']['avg_flesch_ease']:>8.1f}    {agg['readability']['generated']['avg_flesch_ease']:>8.1f}    {agg['readability']['reference']['avg_flesch_ease']:>8.1f}")
    print(f"   Grade Level:         {agg['readability']['original']['avg_grade_level']:>8.1f}    {agg['readability']['generated']['avg_grade_level']:>8.1f}    {agg['readability']['reference']['avg_grade_level']:>8.1f}")
    print(f"   Avg Word Count:      {agg['readability']['original']['avg_word_count']:>8.0f}    {agg['readability']['generated']['avg_word_count']:>8.0f}    {agg['readability']['reference']['avg_word_count']:>8.0f}")
    
    print("\n✨ IMPROVEMENTS (Generated vs Original):")
    print(f"   Flesch Ease Improvement: {agg['improvements']['flesch_ease_improvement']:+.1f} points")
    print(f"   Grade Level Reduction:   {agg['improvements']['grade_level_reduction']:+.1f} grades")
    print(f"   Word Count Reduction:    {agg['improvements']['word_count_reduction_pct']:.1f}%")
    
    print("\n" + "="*70)
    
    # Interpretation guide
    print("\n📚 INTERPRETATION GUIDE:")
    print("   Flesch Reading Ease: Higher = easier (60-70 is plain English)")
    print("   Grade Level: Lower = simpler (6-8 is ideal for general audience)")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the UnArxiv finetuned model")
    parser.add_argument(
        "--test-size", 
        type=int, 
        default=10,
        help="Number of samples to evaluate (default: 10)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/training_data.json",
        help="Path to training data JSON file"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="logs/evaluation_results.json",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--test-split-start",
        type=float,
        default=0.9,
        help="Start of test split (0.9 = last 10% of data)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load test data
        logger.info(f"Loading data from {args.data_path}")
        with open(args.data_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        # Use last portion as test set (wasn't seen during training)
        split_idx = int(len(all_data) * args.test_split_start)
        test_data = all_data[split_idx:]
        logger.info(f"Using {len(test_data)} samples from test split")
        
        # Initialize evaluator
        evaluator = ModelEvaluator()
        evaluator.load_model()
        
        # Run evaluation
        results = evaluator.evaluate(test_data, num_samples=args.test_size)
        
        # Print results
        print_results(results)
        
        # Save results
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        results["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "test_size": args.test_size,
            "data_path": args.data_path,
        }
        
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output_path}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
