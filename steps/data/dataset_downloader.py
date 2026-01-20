import sys
from datasets import load_dataset
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

def download_arxiv_abstracts():
    """
    Downloads the arXiv summarization dataset and extracts abstracts.

    This function utilizes the Hugging Face datasets library to download the
    'ccdv/arxiv-summarization' dataset, specifically the 'document' configuration.
    It retrieves the training split of the dataset and extracts the abstracts
    from each entry.

    Returns:
        list: A list of abstracts extracted from the dataset.
    """
    try:
        logger.info("Starting dataset download...")
        # Load document configuration, train split only
        dataset = load_dataset("ccdv/arxiv-summarization", "document", split="train")
        logger.info("Dataset downloaded successfully.")
        # Extract just the abstracts
        abstracts = dataset['abstract']
        logger.info(f"Extracted {len(abstracts)} abstracts from the dataset.")
        return abstracts
    
    except Exception as e:
        logger.error("An error occurred while downloading the dataset.")
        raise CustomException(str(e), sys) from e

if __name__ == "__main__":
    try:
        abstracts = download_arxiv_abstracts()
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)
