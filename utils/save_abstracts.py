import sys
import os
import json
from utils.logger import get_logger
from utils.custom_exception import CustomException 

logger = get_logger(__name__)

def save_abstracts(abstracts:list, file_path="data/selected_abstracts.json"):
    """
    Save list of abstracts to a JSON file.
    
    Args:
        abstracts (list): The list of selected abstracts to save.
        file_path (str): The file path where the abstracts will be saved.
    """
    try:
        if not isinstance(abstracts, list):
            logger.error("Invalid abstracts format: Expected a list.")
            raise ValueError("Abstracts should be provided as a list.")
        
        logger.info(f"Saving {len(abstracts)} abstracts to {file_path}.")
        
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(abstracts, f, ensure_ascii=False, indent=4)

        logger.info("Abstracts saved successfully.")
    
    except Exception as e:
        logger.error("An error occurred while saving abstracts.")
        raise CustomException(str(e), sys) from e

if __name__ == "__main__":
    from steps.dataset_downloader import download_arxiv_abstracts
    from steps.abstracts_selector import select_abstracts
    from steps.distillation import data_distillation

    abstracts = download_arxiv_abstracts()
    selected_abstracts = select_abstracts(abstracts)
    distilled_abstracts = data_distillation(selected_abstracts)
    save_abstracts(distilled_abstracts, file_path="data/distilled_abstracts.json")