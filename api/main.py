"""
FastAPI backend for the UnArxiv model.

Provides streaming and non-streaming summarization of arXiv abstracts
using a fine-tuned Qwen2.5-3B model.

Endpoints:
    GET  /              - Serve the web frontend
    GET  /api           - API information
    GET  /health        - Health check and model status
    POST /summarize     - Non-streaming summarization
    POST /summarize/stream - Streaming summarization (SSE)
"""

import gc
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from pathlib import Path
from threading import Thread
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from transformers import TextIteratorStreamer
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger
from utils.model_utils import (
    load_finetuned_model,
    build_simplify_prompt,
    prepare_inputs,
    cleanup_memory,
    DEFAULT_MAX_NEW_TOKENS,
)

logger = get_logger(__name__)


# ============================================================================
# GLOBAL STATE
# ============================================================================

# Model components (initialized on startup)
model = None
tokenizer = None
device = None


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class AbstractRequest(BaseModel):
    """
    Request model for abstract summarization.
    
    Attributes:
        abstract: The scientific abstract text to simplify (10-10000 chars)
        max_new_tokens: Maximum tokens to generate (50-2048, default 512)
    """
    abstract: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="The scientific abstract to simplify"
    )
    max_new_tokens: int = Field(
        default=DEFAULT_MAX_NEW_TOKENS,
        ge=50,
        le=2048,
        description="Maximum number of tokens to generate"
    )


class SummarizationResponse(BaseModel):
    """
    Response model for non-streaming summarization.
    
    Attributes:
        simplified_text: The simplified version of the abstract
        tokens_generated: Number of tokens in the response
    """
    simplified_text: str
    tokens_generated: int


# ============================================================================
# APP LIFECYCLE
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for model loading/unloading.
    
    - Startup: Loads the finetuned model in a background thread
    - Shutdown: Cleans up model memory
    """
    global model, tokenizer, device
    
    logger.info("Starting model loading...")
    
    # Load model in executor to not block the event loop
    loop = asyncio.get_event_loop()
    model, tokenizer, device = await loop.run_in_executor(
        None,
        lambda: load_finetuned_model()
    )
    
    logger.info("Model ready for inference")
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down and cleaning up...")
    del model, tokenizer
    cleanup_memory(device)


# ============================================================================
# APP CONFIGURATION
# ============================================================================

app = FastAPI(
    title="UnArxiv API",
    description="Simplify arXiv abstracts using a fine-tuned LLM",
    version="1.0.0",
    lifespan=lifespan
)

# Static files directory for frontend
STATIC_DIR = Path(__file__).parent / "static"

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# GENERATION UTILITIES
# ============================================================================

async def generate_stream(abstract: str, max_new_tokens: int) -> AsyncGenerator[str, None]:
    """
    Generate streaming response using TextIteratorStreamer.
    
    Yields SSE-formatted tokens as they are generated.
    
    Args:
        abstract: The abstract to simplify
        max_new_tokens: Maximum tokens to generate
        
    Yields:
        str: SSE data events with generated tokens
    """
    global model, tokenizer, device
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    # Prepare inputs using shared utility
    messages = build_simplify_prompt(abstract)
    inputs = prepare_inputs(tokenizer, messages, device)
    
    # Create streamer for real-time token output
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    
    # Configure generation
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }
    
    # Run generation in background thread
    def generate():
        with torch.inference_mode():
            model.generate(**generation_kwargs)
    
    thread = Thread(target=generate)
    thread.start()
    
    # Stream tokens as they're generated
    try:
        for text in streamer:
            if text:
                yield f"data: {text}\n\n"
                await asyncio.sleep(0)  # Allow other async tasks
    finally:
        thread.join()
        del inputs
        cleanup_memory(device)
    
    yield "data: [DONE]\n\n"


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Serve the main HTML page.
    
    Returns the frontend if available, otherwise a simple message.
    """
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse(
        content="<h1>UnArxiv API</h1><p>Frontend not found. API is running.</p>"
    )


@app.get("/api")
async def api_info():
    """
    API information endpoint.
    
    Returns available endpoints and API metadata.
    """
    return {
        "name": "UnArxiv API",
        "version": "1.0.0",
        "description": "Simplify arXiv abstracts using a fine-tuned LLM",
        "endpoints": {
            "/": "GET - Web frontend",
            "/api": "GET - This endpoint",
            "/health": "GET - Health check",
            "/summarize": "POST - Non-streaming summarization",
            "/summarize/stream": "POST - Streaming summarization (SSE)",
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns model status and device information.
    """
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
    
    Takes an abstract and returns the complete simplified text.
    
    Args:
        request: AbstractRequest with abstract and max_new_tokens
        
    Returns:
        SummarizationResponse with simplified text and token count
    """
    global model, tokenizer, device
    
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Please wait and try again."
        )
    
    try:
        # Prepare inputs using shared utility
        messages = build_simplify_prompt(request.abstract)
        inputs = prepare_inputs(tokenizer, messages, device)
        input_length = inputs["input_ids"].shape[1]
        
        # Generate
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Extract generated tokens (strip prompt)
        new_token_ids = generated_ids[0][input_length:]
        result = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        
        # Cleanup
        del inputs, generated_ids
        cleanup_memory(device)
        
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
    
    Returns tokens as they are generated for real-time display.
    
    SSE Format:
        data: <token>     - Generated token
        data: [DONE]      - Generation complete
    
    Args:
        request: AbstractRequest with abstract and max_new_tokens
        
    Returns:
        StreamingResponse with SSE events
    """
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Please wait and try again."
        )
    
    return StreamingResponse(
        generate_stream(request.abstract, request.max_new_tokens),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1  # Single worker required for shared model state
    )