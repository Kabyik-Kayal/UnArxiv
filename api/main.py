"""
FastAPI backend for the UnArxiv model.
Provides streaming summarization of arXiv abstracts.
"""

import gc
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from threading import Thread

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Global model variables
model = None
tokenizer = None
device = None


class AbstractRequest(BaseModel):
    """Request model for abstract summarization."""
    abstract: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="The scientific abstract to simplify"
    )
    max_new_tokens: int = Field(
        default=512,
        ge=50,
        le=2048,
        description="Maximum number of tokens to generate"
    )


class SummarizationResponse(BaseModel):
    """Response model for non-streaming summarization."""
    simplified_text: str
    tokens_generated: int


def load_model_sync():
    """Load the finetuned model and tokenizer synchronously."""
    global model, tokenizer, device
    
    device = "xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    logger.info("Moving model to device...")
    base_model = base_model.to(device)
    base_model.eval()
    
    logger.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, "model/qwen-arxiv-simplified-arc")
    model.eval()
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "model/qwen-arxiv-simplified-arc",
        trust_remote_code=True
    )
    
    logger.info("Model loaded successfully!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load model on startup."""
    logger.info("Starting model loading...")
    # Load model in a separate thread to not block
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_model_sync)
    yield
    # Cleanup on shutdown
    global model, tokenizer
    logger.info("Shutting down and cleaning up...")
    del model, tokenizer
    gc.collect()
    if torch.xpu.is_available():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


# Create FastAPI app
app = FastAPI(
    title="UnArxiv API",
    description="Simplify arXiv abstracts using a fine-tuned LLM",
    version="1.0.0",
    lifespan=lifespan
)

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def prepare_input(abstract: str):
    """Prepare input for the model."""
    messages = [
        {
            "role": "user",
            "content": f"Simplify the following scientific abstract into plain language that anyone can understand. Use simple words, short sentences, and everyday analogies.\n\n{abstract}"
        },
    ]
    
    # Build text with template, then tokenize separately
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    return inputs


async def generate_stream(abstract: str, max_new_tokens: int) -> AsyncGenerator[str, None]:
    """Generate streaming response using TextIteratorStreamer."""
    global model, tokenizer, device
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    inputs = prepare_input(abstract)
    
    # Create streamer
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    
    # Generation kwargs
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }
    
    # Run generation in a separate thread
    def generate():
        with torch.no_grad():
            model.generate(**generation_kwargs)
    
    thread = Thread(target=generate)
    thread.start()
    
    # Stream tokens as they're generated
    try:
        for text in streamer:
            if text:
                yield f"data: {text}\n\n"
                await asyncio.sleep(0)  # Allow other tasks to run
    finally:
        thread.join()
        # Cleanup
        del inputs
        if device == "xpu":
            torch.xpu.synchronize()
        gc.collect()
    
    yield "data: [DONE]\n\n"


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse(content="<h1>UnArxiv API</h1><p>Frontend not found. API is running.</p>")


@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "name": "UnArxiv API",
        "version": "1.0.0",
        "description": "Simplify arXiv abstracts using a fine-tuned LLM",
        "endpoints": {
            "/summarize": "POST - Non-streaming summarization",
            "/summarize/stream": "POST - Streaming summarization (SSE)",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global model, tokenizer, device
    return {
        "status": "healthy" if model is not None else "loading",
        "model_loaded": model is not None,
        "device": device if device else "not set"
    }


@app.post("/summarize", response_model=SummarizationResponse)
async def summarize(request: AbstractRequest):
    """
    Non-streaming summarization endpoint.
    Returns the complete simplified text.
    """
    global model, tokenizer, device
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please wait and try again.")
    
    try:
        inputs = prepare_input(request.abstract)
        input_length = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Strip the prompt tokens
        new_token_ids = generated_ids[0][input_length:]
        result = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        
        # Cleanup
        del inputs, generated_ids
        if device == "xpu":
            torch.xpu.synchronize()
        gc.collect()
        
        return SummarizationResponse(
            simplified_text=result.strip(),
            tokens_generated=len(new_token_ids)
        )
        
    except Exception as e:
        logger.error(f"Error during summarization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@app.post("/summarize/stream")
async def summarize_stream(request: AbstractRequest):
    """
    Streaming summarization endpoint using Server-Sent Events (SSE).
    Returns tokens as they are generated.
    
    The response is a stream of SSE events:
    - data: <token> - Generated token
    - data: [DONE] - Generation complete
    """
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please wait and try again.")
    
    return StreamingResponse(
        generate_stream(request.abstract, request.max_new_tokens),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1  # Single worker for model loading
    )
