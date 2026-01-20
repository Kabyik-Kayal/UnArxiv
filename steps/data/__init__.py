"""
Data Preparation Steps.

This subpackage handles all data acquisition and preparation:
- dataset_downloader: Downloads arXiv abstracts from HuggingFace
- abstracts_selector: Random sampling of abstracts for training
- distillation: Teacher model API calls via Groq (Kimi K2)
- training_data: Formats data for instruction tuning

Usage:
    python -m pipelines.data_preparation
"""

__all__ = [
    "download_arxiv_abstracts",
    "select_abstracts",
    "data_distillation",
    "TrainingDataGenerator",
]


def __getattr__(name):
    """Lazy import to avoid RuntimeWarning when running modules directly."""
    if name == "download_arxiv_abstracts":
        from steps.data.dataset_downloader import download_arxiv_abstracts
        return download_arxiv_abstracts
    elif name == "select_abstracts":
        from steps.data.abstracts_selector import select_abstracts
        return select_abstracts
    elif name == "data_distillation":
        from steps.data.distillation import data_distillation
        return data_distillation
    elif name == "TrainingDataGenerator":
        from steps.data.training_data import TrainingDataGenerator
        return TrainingDataGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
