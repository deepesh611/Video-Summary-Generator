
import modal

# Define the image with all dependencies and local source code
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsm6", "libxext6")
    .pip_install(
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "python-dotenv>=1.0.0",
        "python-multipart>=0.0.6",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
    )
    # Mount the backend directory to access main.py and pipeline.py
    .add_local_dir("backend", "/backend")
)

# Create the Modal app
app = modal.App("video-summary-generator")

# Define the ASGI web endpoint
@app.function(
    image=image,
    # Use GPU for faster processing (remove if you want CPU-only)
    # Set to None for CPU-only: gpu=None
    gpu="T4",  # Can be "T4", "A10G", "A100", or None for CPU
    # Increase timeout for long video processing
    timeout=1800,  # 30 minutes
    # Set memory limit
    memory=8192,  # 8GB RAM
    # Allow external traffic
    allow_concurrent_inputs=2,
    # Load Modal secrets (optional - for API_URL, API_KEY, etc.)
    secrets=[modal.Secret.from_name("env")],
)
@modal.asgi_app()
def fastapi_app():
    """Mount the FastAPI app as an ASGI application."""
    import sys
    import os
    from pathlib import Path
    
    # Add backend to path
    sys.path.insert(0, "/backend")
    
    # Set working directory
    os.chdir("/backend")
    
    # Environment variables are automatically loaded from Modal secrets
    # if you created a secret named "env" with API_URL, API_KEY, etc.
    # Modal secrets are already in os.environ at this point
    
    try:
        # Import and return the FastAPI app
        from main import app as fastapi_application
        return fastapi_application
    except Exception as e:
        # Print error for debugging
        import traceback
        print(f"Error importing FastAPI app: {e}")
        print(traceback.format_exc())
        raise

