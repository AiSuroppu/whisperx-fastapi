import os
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from loguru import logger

from whisperx_fastapi.api.endpoints import router as api_router
from whisperx_fastapi.core.config import settings

# Initialize FastAPI app
app = FastAPI(
    title="WhisperX-FastAPI Service",
    version="1.0.0",
    description="A high-performance API for ASR, alignment, and diarization using whisperx."
)

# --- Exception Handlers ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handles errors in request validation (e.g., multipart form data)."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Request Validation Error", "details": exc.errors()},
    )

@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    """Handles errors in Pydantic model validation from request bodies."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "JSON Body Validation Error", "details": exc.errors()},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"An unhandled exception occurred: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal Server Error", "details": str(exc)},
    )

# --- Logging Configuration ---
os.makedirs("logs", exist_ok=True)
logger.add("logs/{time}.log", level=settings.LOG_LEVEL.upper(), rotation="10 MB", compression="zip")

# --- Include Routers ---
app.include_router(api_router, prefix="/api/v1")

@app.get("/", include_in_schema=False)
def read_root():
    return {"message": "Welcome to the WhisperX-FastAPI service. See /docs for API details."}


if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True, workers=1)