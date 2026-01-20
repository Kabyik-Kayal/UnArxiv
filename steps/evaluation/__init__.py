"""
Evaluation Steps.

This subpackage handles metrics computation and model comparison:
- compute_metrics: ROUGE scores and readability metrics
- test_models: Quick comparison between base and finetuned models

Usage:
    python -m steps.evaluation.compute_metrics
    python -m steps.evaluation.test_models
"""

# Lazy imports - only import when explicitly accessed
# This prevents circular import issues when running modules directly

__all__ = [
    "compute_rouge",
    "compute_readability",
    "compare_results",
]


def __getattr__(name):
    """Lazy import to avoid RuntimeWarning when running modules directly."""
    if name in ("compute_rouge", "compute_readability", "compare_results"):
        from steps.evaluation.compute_metrics import (
            compute_rouge,
            compute_readability,
            compare_results,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
