import sys
from datasets import load_dataset
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

def download_arxiv_abstracts():
    """
    Downloads a sample of arXiv abstracts from the Hugging Face dataset.
    
    This function utilizes the Hugging Face datasets library to stream the
    'ccdv/arxiv-summarization' dataset, shuffles it, and extracts a fixed
    number of abstracts from the training split.

    Returns:
        list: A list of abstracts extracted from the dataset.

    Raises:
        CustomException: If an error occurs during the download or extraction process.
    """
    try:
        logger.info("Starting dataset download (streaming mode with random sampling)...")
        dataset = load_dataset("ccdv/arxiv-summarization", "document", split="train", streaming=True)
        dataset = dataset.shuffle(seed=42, buffer_size=10000)
        num_samples = 1000
        abstracts = []
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
            abstracts.append(example['abstract'])
            if (i + 1) % 500 == 0:
                logger.info(f"Downloaded {i + 1}/{num_samples} abstracts...")
        
        logger.info(f"Successfully extracted {len(abstracts)} abstracts from the dataset.")
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
