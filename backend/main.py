"""
FastAPI backend for video-to-summary pipeline.

This module provides the API endpoints for uploading videos and processing them
through the deep learning pipeline to generate summaries.
"""

import os
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pipeline import run_pipeline

# Load environment variables from backend/.env
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Initialize FastAPI app
app = FastAPI(
    title="Video Summary Generator API",
    description="API for processing videos and generating summaries",
    version="1.0.0"
)

# Configure CORS (allow frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
UPLOADS_DIR = Path(__file__).parent / "uploads"
SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".mov"}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB limit


def ensure_uploads_directory():
    """Create uploads directory if it doesn't exist."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def generate_run_id() -> str:
    """
    Generate a unique run ID using UUID.
    
    Returns:
        str: A unique identifier for this processing run
    """
    return str(uuid.uuid4())


def get_file_extension(filename: str) -> Optional[str]:
    """
    Extract file extension from filename.
    
    Args:
        filename: The name of the file
        
    Returns:
        Optional[str]: The file extension (with dot) or None if invalid
    """
    return Path(filename).suffix.lower()


def is_valid_video_format(filename: str) -> bool:
    """
    Check if the file has a supported video format.
    
    Args:
        filename: The name of the file to check
        
    Returns:
        bool: True if format is supported, False otherwise
    """
    ext = get_file_extension(filename)
    return ext in SUPPORTED_EXTENSIONS if ext else False


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    ensure_uploads_directory()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Video Summary Generator API is running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "uploads_directory": str(UPLOADS_DIR),
        "uploads_directory_exists": UPLOADS_DIR.exists()
    }


@app.post("/process_upload")
async def process_upload(file: UploadFile = File(...)):
    """
    Process an uploaded video file and generate a summary.
    
    This endpoint:
    1. Accepts a video file as multipart/form-data
    2. Creates a unique run ID
    3. Saves the video to backend/uploads/{run_id}/input_video.{ext}
    4. Runs the pipeline to generate a summary
    5. Returns the summary and run_id
    
    Args:
        file: The uploaded video file
        
    Returns:
        JSONResponse: Contains summary text and run_id
        
    Raises:
        HTTPException: If file validation fails or processing error occurs
    """
    try:
        # Validate file format
        if not is_valid_video_format(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
            )
        
        # Generate unique run ID
        run_id = generate_run_id()
        
        # Create run directory
        run_dir = UPLOADS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Get file extension
        file_ext = get_file_extension(file.filename)
        
        # Save uploaded file
        input_video_path = run_dir / f"input_video{file_ext}"
        
        # Read file content
        file_content = await file.read()
        
        # Check file size
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE / (1024 * 1024):.0f} MB"
            )
        
        # Write file to disk
        with open(input_video_path, "wb") as f:
            f.write(file_content)
        
        # Run the pipeline to generate summary
        # Note: This may take some time depending on video length and processing complexity
        try:
            summary = run_pipeline(str(input_video_path))
            
            # Validate that pipeline returned a summary
            if not summary or not isinstance(summary, str):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Pipeline did not return a valid summary"
                )
            
        except Exception as pipeline_error:
            # Log the error (without exposing sensitive information)
            error_msg = str(pipeline_error)
            # Remove any potential API keys or secrets from error message
            # This is a basic filter - enhance as needed
            if "API_KEY" in error_msg or "api_key" in error_msg.lower():
                error_msg = "Pipeline processing error (credentials filtered)"
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error during pipeline processing: {error_msg}"
            )
        
        # Return success response with summary and run_id
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "run_id": run_id,
                "summary": summary,
                "original_filename": file.filename,
                "file_size_bytes": len(file_content)
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    
    except Exception as e:
        # Catch any unexpected errors
        error_msg = str(e)
        # Filter out sensitive information
        if "API_KEY" in error_msg or "api_key" in error_msg.lower():
            error_msg = "Processing error (sensitive information filtered)"
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {error_msg}"
        )


# Future endpoints can be added here:
# - GET /status/{run_id} - Check processing status
# - GET /results/{run_id} - Get processing results
# - POST /batch_process - Process multiple videos
# - DELETE /uploads/{run_id} - Clean up specific upload

if __name__ == "__main__":
    import uvicorn
    
    # Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
