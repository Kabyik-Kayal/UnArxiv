import sys
import random
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

def select_abstracts(abstracts, num_abstracts=1000, seed=42):
    """
    Selects a random sample of abstracts from the provided list.

    This function randomly selects a specified number of abstracts from the
    input list to be used for further processing or distillation.

    Args:
        abstracts (list): The list of abstracts to sample from.
        num_abstracts (int): The number of abstracts to select. Default is 1000.
    
    Returns:
        selected_abstracts (list): A list containing the selected abstracts.
    """
    try:
        random.seed(seed)
        abstract_indices = random.sample(range(len(abstracts)), num_abstracts)
        selected_abstracts = [abstracts[i] for i in abstract_indices]
        logger.info(f"Selected {len(selected_abstracts)} abstracts successfully for distillation.")
        logger.info(f"Sample selected abstracts: {selected_abstracts[9:10]}")
        return selected_abstracts
    except Exception as e:
        logger.error("An error occurred while selecting abstracts.")
        raise CustomException(str(e), sys) from e
    

if __name__ == "__main__":
    try:
        from steps.dataset_downloader import download_arxiv_abstracts
        abstracts = download_arxiv_abstracts()
        selected_abstracts = select_abstracts(abstracts)
    except Exception as e:
        logger.error(f"Selection failed: {e}")
        sys.exit(1)