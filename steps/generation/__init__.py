"""
Output Generation Steps.

This subpackage handles output generation for inference and evaluation:
- generate_outputs: Unified generator for base and finetuned models
- inference: Standalone inference script with progress display

Usage:
    python -m steps.generation.generate_outputs --model-type base
    python -m steps.generation.inference
"""

__all__ = [
    "load_model_with_progress",
]


def __getattr__(name):
    """Lazy import to avoid RuntimeWarning when running modules directly."""
    if name == "load_model_with_progress":
        from steps.generation.inference import load_model_with_progress
        return load_model_with_progress
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
