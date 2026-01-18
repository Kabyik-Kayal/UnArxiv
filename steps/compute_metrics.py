"""
Compute evaluation metrics comparing base vs finetuned model outputs.
Loads pre-generated outputs from JSON files.
"""

import json
import sys
import os
import argparse
from datetime import datetime
from typing import Dict, List

from rouge_score import rouge_scorer
import textstat

from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)


def compute_rouge(prediction: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1": round(scores['rouge1'].fmeasure, 4),
        "rouge2": round(scores['rouge2'].fmeasure, 4),
        "rougeL": round(scores['rougeL'].fmeasure, 4),
    }


def compute_readability(text: str) -> Dict[str, float]:
    """Compute Flesch reading ease and grade level."""
    if not text or len(text.split()) < 5:
        return {"flesch_reading_ease": 0, "grade_level": 0, "word_count": 0}
    
    return {
        "flesch_reading_ease": round(textstat.flesch_reading_ease(text), 2),
        "grade_level": round(textstat.flesch_kincaid_grade(text), 2),
        "word_count": textstat.lexicon_count(text, removepunct=True),
    }


def is_valid_output(text: str, name: str, sample_idx: int) -> bool:
    """Check if output is valid (not corrupted)."""
    words = len(text.split())
    if words < 15:
        logger.warning(f"Sample {sample_idx} {name} too short: {words} words")
        return False
    
    corrupt_patterns = ["userassistant", "usereditor", "userdivider", "<|im_start|>", "<|im_end|>"]
    for pattern in corrupt_patterns:
        if pattern in text.lower():
            logger.warning(f"Sample {sample_idx} {name} contains corruption pattern: {pattern}")
            return False
    
    return True


def compare_results(
    abstracts: List[str],
    references: List[str],
    base_outputs: List[str],
    finetuned_outputs: List[str]
) -> Dict:
    """Compare base vs finetuned outputs using ROUGE and readability metrics."""
    
    results = {"samples": [], "summary": {}}
    
    # Accumulators for averages
    base_metrics = {"rouge1": 0, "rouge2": 0, "rougeL": 0, "flesch": 0, "grade": 0}
    ft_metrics = {"rouge1": 0, "rouge2": 0, "rougeL": 0, "flesch": 0, "grade": 0}
    valid_count = 0
    
    for i, (abstract, ref, base_out, ft_out) in enumerate(zip(abstracts, references, base_outputs, finetuned_outputs)):
        logger.info(f"Sample {i+1} base words: {len(base_out.split())}, ft words: {len(ft_out.split())}")
        
        base_valid = is_valid_output(base_out, "Base", i+1)
        ft_valid = is_valid_output(ft_out, "Finetuned", i+1)
        
        if not base_valid or not ft_valid:
            logger.warning(f"Sample {i+1}: Skipping due to invalid output")
            continue
        
        valid_count += 1
        
        # ROUGE scores vs reference
        base_rouge = compute_rouge(base_out, ref)
        ft_rouge = compute_rouge(ft_out, ref)
        
        # Readability metrics
        base_read = compute_readability(base_out)
        ft_read = compute_readability(ft_out)
        
        # Accumulate
        for k in ["rouge1", "rouge2", "rougeL"]:
            base_metrics[k] += base_rouge[k]
            ft_metrics[k] += ft_rouge[k]
        base_metrics["flesch"] += base_read["flesch_reading_ease"]
        base_metrics["grade"] += base_read["grade_level"]
        ft_metrics["flesch"] += ft_read["flesch_reading_ease"]
        ft_metrics["grade"] += ft_read["grade_level"]
        
        results["samples"].append({
            "id": i + 1,
            "original": abstract[:150] + "...",
            "reference": ref[:150] + "...",
            "base_output": base_out[:200] + "..." if len(base_out) > 200 else base_out,
            "finetuned_output": ft_out[:200] + "..." if len(ft_out) > 200 else ft_out,
            "base_rouge": base_rouge,
            "finetuned_rouge": ft_rouge,
            "base_readability": base_read,
            "finetuned_readability": ft_read,
        })
    
    # Compute averages
    n = valid_count or 1
    results["summary"] = {
        "valid_samples": valid_count,
        "base_model": {
            "avg_rouge1": round(base_metrics["rouge1"] / n, 4),
            "avg_rouge2": round(base_metrics["rouge2"] / n, 4),
            "avg_rougeL": round(base_metrics["rougeL"] / n, 4),
            "avg_flesch_reading_ease": round(base_metrics["flesch"] / n, 2),
            "avg_grade_level": round(base_metrics["grade"] / n, 2),
        },
        "finetuned_model": {
            "avg_rouge1": round(ft_metrics["rouge1"] / n, 4),
            "avg_rouge2": round(ft_metrics["rouge2"] / n, 4),
            "avg_rougeL": round(ft_metrics["rougeL"] / n, 4),
            "avg_flesch_reading_ease": round(ft_metrics["flesch"] / n, 2),
            "avg_grade_level": round(ft_metrics["grade"] / n, 2),
        },
        "improvement": {
            "rouge1_delta": round((ft_metrics["rouge1"] - base_metrics["rouge1"]) / n, 4),
            "rouge2_delta": round((ft_metrics["rouge2"] - base_metrics["rouge2"]) / n, 4),
            "rougeL_delta": round((ft_metrics["rougeL"] - base_metrics["rougeL"]) / n, 4),
            "flesch_delta": round((ft_metrics["flesch"] - base_metrics["flesch"]) / n, 2),
            "grade_delta": round((base_metrics["grade"] - ft_metrics["grade"]) / n, 2),
        }
    }
    
    return results


def print_summary(results: Dict):
    """Print comparison summary to console."""
    s = results["summary"]
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS: BASE MODEL vs FINETUNED MODEL")
    print("=" * 70)
    print(f"Valid samples evaluated: {s['valid_samples']}")
    
    print("\n📊 ROUGE SCORES (vs reference):")
    print(f"{'Metric':<10} {'Base':>12} {'Finetuned':>12} {'Delta':>12}")
    print("-" * 50)
    for k in ["rouge1", "rouge2", "rougeL"]:
        base = s["base_model"][f"avg_{k}"]
        ft = s["finetuned_model"][f"avg_{k}"]
        delta = s["improvement"][f"{k}_delta"]
        print(f"{k:<10} {base:>12.4f} {ft:>12.4f} {delta:>+12.4f}")
    
    print("\n📖 READABILITY METRICS:")
    print(f"{'Metric':<20} {'Base':>12} {'Finetuned':>12} {'Delta':>12}")
    print("-" * 60)
    print(f"{'Flesch Reading Ease':<20} {s['base_model']['avg_flesch_reading_ease']:>12.2f} {s['finetuned_model']['avg_flesch_reading_ease']:>12.2f} {s['improvement']['flesch_delta']:>+12.2f}")
    print(f"{'Grade Level':<20} {s['base_model']['avg_grade_level']:>12.2f} {s['finetuned_model']['avg_grade_level']:>12.2f} {s['improvement']['grade_delta']:>+12.2f}")
    
    print("\n📚 INTERPRETATION:")
    print("   ROUGE: Higher = better match to reference (+ delta = finetuning helped)")
    print("   Flesch: Higher = easier to read (60-70 is plain English)")
    print("   Grade: Lower = simpler (+ delta = grade level reduced)")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compute evaluation metrics")
    parser.add_argument("--base-outputs", type=str, default="logs/base_outputs.json")
    parser.add_argument("--finetuned-outputs", type=str, default="logs/finetuned_outputs.json")
    parser.add_argument("--output-path", type=str, default="logs/evaluation_results.json")
    args = parser.parse_args()
    
    try:
        # Load outputs
        logger.info(f"Loading base outputs from {args.base_outputs}")
        with open(args.base_outputs, 'r', encoding='utf-8') as f:
            base_data = json.load(f)
        
        logger.info(f"Loading finetuned outputs from {args.finetuned_outputs}")
        with open(args.finetuned_outputs, 'r', encoding='utf-8') as f:
            finetuned_data = json.load(f)
        
        # Extract data
        abstracts = base_data["abstracts"]
        references = base_data["references"]
        base_outputs = base_data["outputs"]
        finetuned_outputs = finetuned_data["outputs"]
        
        # Compare
        logger.info("Computing metrics...")
        results = compare_results(abstracts, references, base_outputs, finetuned_outputs)
        results["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "base_outputs_file": args.base_outputs,
            "finetuned_outputs_file": args.finetuned_outputs,
        }
        
        # Print and save
        print_summary(results)
        
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_path}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
