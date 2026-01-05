import os
import sys
import time
from dotenv import load_dotenv
from groq import Groq
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY environment variable is not set.")
    raise ValueError("GROQ_API_KEY environment variable is not set")
    
def data_distillation(abstracts:list, api_key:str=GROQ_API_KEY):
    """
    This function performs data distillation on the selected abstracts 
    by sending them to a teacher model (eg. Kimi K2) via the Groq API.
    The teacher model simplifies the abstracts and returns the distilled versions,
    which will later be used to fine-tune our student model (Qwen 2.5 3B instruct).

    Args:
        abstracts (list): The list of selected abstracts to distill.
        api_key (str): The API key for the Groq API.
    
    Returns:
        distilled_abstracts (list): A list containing the distilled abstracts.
            If an API error occurs, only successfully distilled abstracts are returned.
            Requests are spaced with a 2 second delay to avoid throttling.
    """
    
    try:
        client = Groq(api_key=api_key)
        
        logger.info(f"Starting data distillation for {len(abstracts)} abstracts...")

        distilled_abstracts = []

        for idx, abstract in enumerate(abstracts, start=1):
            if idx > 1:
                time.sleep(5)
            try:
                completion = client.chat.completions.create(
                    model="moonshotai/kimi-k2-instruct-0905",
                    messages=[
                        {
                            "role": "user",
                            "content": f"""Rewrite the research abstract below so a first-year undergraduate (any major) can understand it while keeping the original meaning.
                                    
                                    Audience + tone:
                                    - Curious, friendly, and engaging (no hype, no overselling).
                                    - Assume minimal background knowledge.

                                    Hard constraints:
                                    - Maximum 200 words total.
                                    - Do not add facts, results, numbers, citations, or claims that are not in the original.
                                    - Preserve the main problem, approach, key findings (if present), and implication.

                                    Clarity rules:
                                    - Replace jargon with plain language whenever possible.
                                    - If a technical term is necessary, define it immediately in simple words.
                                    - Use 1–2 concrete analogies or real-world examples to explain the central idea (only if they fit naturally).
                                    - Explicitly explain why this research matters (impact, usefulness, or what it helps understand/solve), not just what was done.
                                    - Prefer short sentences and active voice.

                        Original Abstract: {abstract}
                        """
                        }
                    ],
                    temperature=0.6,
                    max_completion_tokens=4096,
                    top_p=1,
                    stream=True,
                    stop=None,
                )
                distilled_content = ""
                for chunk in completion:
                    delta = chunk.choices[0].delta
                    piece = getattr(delta, "content", None)
                    if piece:
                        distilled_content += piece
            except Exception as error:
                logger.error("Error distilling abstract %d: %s", idx, error)
                logger.info("Returning %d distilled abstracts so far.", len(distilled_abstracts))
                break
            distilled_abstracts.append(distilled_content)
            logger.info("Distilled %d abstracts successfully so far...", len(distilled_abstracts))
        if len(distilled_abstracts) == len(abstracts):
            logger.info("Data distillation completed successfully.")
        else:
            logger.warning("Data distillation stopped after %d abstracts due to errors.", len(distilled_abstracts))
        logger.info(f"Sample distilled abstract: {distilled_abstracts[0:1]}")
        return distilled_abstracts
    
    except Exception as e:
        logger.error("An error occurred during data distillation.")
        raise CustomException(str(e), sys) from e
    
if __name__ == "__main__":
    try:
        from steps.dataset_downloader import download_arxiv_abstracts
        from steps.abstracts_selector import select_abstracts
        abstracts = download_arxiv_abstracts()
        selected_abstracts = select_abstracts(abstracts)
        distilled_abstracts = data_distillation(selected_abstracts[:1])
    except Exception as e:
        logger.error(f"Distillation failed: {e}")
        sys.exit(1)