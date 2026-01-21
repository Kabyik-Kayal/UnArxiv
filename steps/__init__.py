"""
UnArxiv Steps Package.

This package contains modular step implementations organized by pipeline phase:
- data/       : Data acquisition and preparation (download, select, distill, format)
- training/   : Model finetuning with LoRA
- generation/ : Output generation for inference and evaluation
- evaluation/ : Metrics computation and model comparison

Usage:
    # Run individual steps
    python -m steps.training.finetuning
    python -m steps.generation.inference
    python -m steps.evaluation.compute_metrics
    
    # Or use the pipeline orchestrators
    python -m pipelines.data_preparation
    python -m pipelines.evaluation_pipeline
"""

__all__ = [
    # Data
    "download_arxiv_abstracts",
    "data_distillation",
    "TrainingDataGenerator",
    # Generation
    "load_model_with_progress",
]


def __getattr__(name):
    """Lazy import to avoid RuntimeWarning when running modules directly."""
    if name == "download_arxiv_abstracts":
        from steps.data.dataset_downloader import download_arxiv_abstracts
        return download_arxiv_abstracts
    elif name == "data_distillation":
        from steps.data.distillation import data_distillation
        return data_distillation
    elif name == "TrainingDataGenerator":
        from steps.data.training_data import TrainingDataGenerator
        return TrainingDataGenerator
    elif name == "load_model_with_progress":
        from steps.generation.inference import load_model_with_progress
        return load_model_with_progress
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
