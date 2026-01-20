"""
Model Training Steps.

This subpackage handles model finetuning:
- finetuning: LoRA training on Intel XPU with memory optimizations

Usage:
    python -m steps.training.finetuning
"""

__all__ = [
    "run_finetuning",
]


def __getattr__(name):
    """Lazy import to avoid RuntimeWarning when running modules directly."""
    if name == "run_finetuning":
        from steps.training.finetuning import main as run_finetuning
        return run_finetuning
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
