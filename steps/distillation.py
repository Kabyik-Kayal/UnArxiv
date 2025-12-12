import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logger.error("API_KEY environment variable is not set.")
    raise ValueError("API_KEY environment variable is not set")
    
def data_distillation(abstracts:list, api_key:str=API_KEY):
    """
    This function performs data distillation on the selected abstracts 
    by sending them to a teacher model (eg. Kimi K2) via an OpenRouter API.
    The teacher model simplifies the abstracts and returns the distilled versions,
    which will later be used to fine-tune our student model (Qwen 2.5 3B instruct).

    Args:
        abstracts (list): The list of selected abstracts to distill.
        api_key (str): The API key for OpenRouter access.
    
    Returns:
        distilled_abstracts (list): A list containing the distilled abstracts.
    """
    
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        logger.info(f"Starting data distillation for {len(abstracts)} abstracts...")

        i = 0
        distilled_abstracts = []

        for abstract in abstracts:
            completion = client.chat.completions.create(
                extra_body={"reasoning": {"enabled": True}},
                model="openai/gpt-oss-120b:free",
                messages=[
                    {
                    "role": "user",
                    "content": f"""You are an expert science communicator. Rewrite the following research paper abstract for a first-year undergraduate student. Rules:
                    - Avoid jargon. If you must use technical terms, explain them.
                    - Use analogies and real-world examples.
                    - Explain WHY this research matters (not just WHAT it does).
                    - Keep it under 200 words.
                    - Make it engaging and curious-inducing.

                    Original Abstract: {abstract}
                    """
                    }
                ]
            )
            distilled_abstracts.append(completion.choices[0].message.content)
            i+=1
            logger.info(f"Distilled {i} abstracts successfully so far...")
        logger.info("Data distillation completed successfully.")
        logger.info(f"Sample distilled abstract: {distilled_abstracts[0:1]}")
        return distilled_abstracts
    
    except Exception as e:
        logger.error("An error occurred during data distillation.")
        raise CustomException(str(e), sys) from e
    
if __name__ == "__main__":
    from steps.dataset_downloader import download_arxiv_abstracts
    from steps.abstracts_selector import select_abstracts
    abstracts = download_arxiv_abstracts()
    selected_abstracts = select_abstracts(abstracts)
    distilled_abstracts = data_distillation(selected_abstracts)