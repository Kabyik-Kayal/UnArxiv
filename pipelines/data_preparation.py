from utils.logger import get_logger
from utils.custom_exception import CustomException
from utils.save_abstracts import save_abstracts
import sys

from steps.dataset_downloader import download_arxiv_abstracts
from steps.abstracts_selector import select_abstracts
from steps.distillation import data_distillation
from steps.training_data import TrainingDataGenerator

logger = get_logger(__name__)
generator = TrainingDataGenerator()

def data_preparation():
    """
    Complete data preparation pipeline:
    1. Download abstracts.
    2. Select a subset of abstracts.
    3. Save selected abstracts.
    4. Distill selected abstracts.
    5. Save distilled abstracts.
    6. Create Training data combining the distilled abstracts with selected abstracts.
    """
    logger.info("Starting data preparation pipeline...")
    abstracts = download_arxiv_abstracts()
    selected_abstracts = select_abstracts(abstracts)
    save_abstracts(selected_abstracts, file_path="data/selected_abstracts.json")
    distilled_abstracts = data_distillation(selected_abstracts)
    save_abstracts(distilled_abstracts, file_path="data/distilled_abstracts.json")
    generator.create_training_data()
    logger.info("Data preparation pipeline completed successfully.")

if __name__ == "__main__":
    data_preparation()