# Video Summary Generator 🎬

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Modal](https://img.shields.io/badge/Modal-Serverless-purple.svg)

A deep learning-based application that generates text summaries from video inputs. The project features a **Streamlit** frontend for video upload and a **FastAPI** backend deployed on **Modal** that processes videos through a sophisticated ML pipeline to generate summaries.

## 🎯 Features

- **🎥 Video Upload Interface**: User-friendly Streamlit frontend supporting `.mp4`, `.mkv`, and `.mov` formats
- **🧠 Intelligent Frame Extraction**: Automatically extracts and selects key frames from videos
- **🤖 Deep Learning Pipeline**:
  - Frame feature extraction using **ResNet50**
  - Frame importance scoring with **LSTM** model
  - Key frame selection based on importance thresholds
  - Caption generation using **BLIP** (Salesforce/blip-image-captioning-base)
  - Optional API-based final summarization
- **☁️ Cloud Deployment**: Backend hosted on **Modal** (serverless with GPU support), frontend on **Streamlit Cloud**
- **🧹 Automatic Cleanup**: Temporary files and old uploads are automatically managed
- **🔒 Secure Configuration**: Environment-based configuration for API keys and secrets

## 🏗️ Architecture

```
┌─────────────────┐          ┌──────────────────┐          ┌─────────────────┐
│  Streamlit      │          │   Modal          │          │   ML Pipeline   │
│  Frontend       │────────▶│   Backend         │────────▶│   (PyTorch)     │
│  (Cloud/Local)  │  HTTP    │   (FastAPI)      │          │   • ResNet50    │
│                 │          │   • GPU Support  │          │   • LSTM        │
│                 │          │   • Auto-scaling │          │   • BLIP        │
└─────────────────┘          └──────────────────┘          └─────────────────┘
```

### Frontend (`app/`)
- **Streamlit Application**: Web-based UI for video upload and result display
- Handles file upload, displays processing status, and manages user sessions
- Automatic cleanup of old uploads (24+ hours)
- Configurable backend API URL (Modal deployment)

### Backend (`backend/`)
- **FastAPI Server**: RESTful API for video processing
- **Modal Deployment**: Serverless deployment with GPU support (T4/A10G)
- **ML Pipeline**: Complete deep learning workflow for video-to-text summarization
  - Video frame extraction (OpenCV)
  - Feature extraction (ResNet50)
  - Frame importance scoring (LSTM)
  - Caption generation (BLIP)
  - Optional API summarization

## 🛠️ Technology Stack

### Core Framework
- ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) **Python 3.11+**
- ![FastAPI](https://img.shields.io/badge/-FastAPI-009688?logo=fastapi&logoColor=white) **FastAPI** - Modern, fast web framework
- ![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?logo=streamlit&logoColor=white) **Streamlit** - Rapid web app development

### Deployment & Infrastructure
- ![Modal](https://img.shields.io/badge/-Modal-000000?logo=modal&logoColor=white) **Modal** - Serverless GPU platform for backend
- ![Streamlit Cloud](https://img.shields.io/badge/-Streamlit_Cloud-FF4B4B?logo=streamlit&logoColor=white) **Streamlit Cloud** - Hosting for frontend

### Deep Learning & ML
- ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=pytorch&logoColor=white) **PyTorch** - Deep learning framework
- ![Torchvision](https://img.shields.io/badge/-Torchvision-EE4C2C?logo=pytorch&logoColor=white) **Torchvision** - Computer vision utilities
- ![Transformers](https://img.shields.io/badge/-Transformers-FFD700?logo=huggingface&logoColor=white) **Transformers (HuggingFace)** - Pre-trained models (BLIP)

### Computer Vision
- ![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?logo=opencv&logoColor=white) **OpenCV** - Video processing and frame extraction
- ![Pillow](https://img.shields.io/badge/-Pillow-013243?logo=python&logoColor=white) **Pillow (PIL)** - Image processing

### Utilities
- ![Uvicorn](https://img.shields.io/badge/-Uvicorn-45948F?logo=python&logoColor=white) **Uvicorn** - ASGI server
- ![python-dotenv](https://img.shields.io/badge/-python--dotenv-3776AB?logo=python&logoColor=white) **python-dotenv** - Environment variable management
- ![Requests](https://img.shields.io/badge/-Requests-3776AB?logo=python&logoColor=white) **Requests** - HTTP client

## 📋 Requirements

- **Python**: 3.8+
- **GPU**: Optional but recommended for faster processing (T4 available on Modal)
- **Memory**: Minimum 4GB RAM (8GB+ recommended for local development)
- **Disk Space**: For video storage and model cache (~5-10GB for models)

## 🚀 Quick Start

### Local Development Setup

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd Video-Summary-Generator
```

#### 2. Create Virtual Environment

```bash
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On Linux/Mac
source .venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Run Locally

**Backend (local):**
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend (local):**
```bash
cd app
# Set backend URL (for local backend)
export BACKEND_API_URL=http://localhost:8000  # Linux/Mac
# OR
set BACKEND_API_URL=http://localhost:8000  # Windows CMD
# OR
$env:BACKEND_API_URL="http://localhost:8000"  # Windows PowerShell

streamlit run main.py
```

## ☁️ Cloud Deployment

### Deploy Backend to Modal

1. **Install Modal CLI:**
```bash
pip install modal
modal token new  # Authenticate
```

2. **Deploy Backend:**
```bash
modal deploy backend/modal_app.py
```

3. **Get Your Modal URL:**
After deployment, Modal provides a URL like:
```
https://your-username--video-summary-generator-fastapi-app.modal.run
```

4. **Configure Secrets (Optional):**
```bash
modal secret create env \
  API_URL=https://your-api-url.com \
  API_KEY=your_api_key \
  MODEL_NAME=tngtech/deepseek-r1t2-chimera:free
```

### Deploy Frontend to Streamlit Cloud

1. **Push code to GitHub**

2. **Configure Streamlit Cloud:**
   - Go to [streamlit.io/cloud](https://streamlit.io/cloud)
   - Click "New app"
   - Connect your GitHub repository
   - Set **Main file path**: `app/main.py`

3. **Add Secrets:**
   In Streamlit Cloud settings, add:
   ```toml
   BACKEND_API_URL = "https://your-modal-url.modal.run"
   ```

4. **Deploy!**

### Test Modal Backend Locally

You can run the frontend locally and connect it to your Modal backend:

```bash
# Set Modal backend URL
export BACKEND_API_URL="https://your-username--video-summary-generator-fastapi-app.modal.run"

# Run frontend
cd app
streamlit run main.py
```

## 📖 Usage

### Using the Application

1. **Upload Video**: Use the Streamlit frontend to upload a video file (`.mp4`, `.mkv`, or `.mov`)
2. **Process**: Click "Process Video" button
3. **Wait**: Processing can take 5-30 minutes depending on video length (first request loads models)
4. **View Results**: The generated summary will be displayed

### API Usage (Direct)

You can also call the Modal API directly:

```bash
curl -X POST "https://your-modal-url.modal.run/process_upload" \
  -H "accept: application/json" \
  -F "file=@your_video.mp4"
```

**Response:**
```json
{
  "status": "success",
  "run_id": "uuid-here",
  "summary": "Generated summary text...",
  "original_filename": "your_video.mp4",
  "file_size_bytes": 12345678
}
```

## 📁 Project Structure

```
Video-Summary-Generator/
│
├── app/                          # Streamlit Frontend
│   ├── main.py                 # Main Streamlit application
│   ├── uploads/                # User uploads (auto-created, gitignored)
│   └── .streamlit/             # Streamlit configuration
│       └── secrets.toml        # Secrets for local (gitignored)
│
├── backend/                     # FastAPI Backend
│   ├── main.py                 # FastAPI application
│   ├── pipeline.py             # ML pipeline implementation
│   ├── modal_app.py            # Modal deployment configuration
│   ├── uploads/                # Processed videos (gitignored)
│   └── .env                    # Environment variables (gitignored)
│
├── notebooks/                   # Development notebooks
│   └── notebook1.ipynb         # Original pipeline development
│
├── requirements.txt            # Root-level dependencies
├── .gitignore                  # Git ignore rules
├── LICENSE                     # License file
└── README.md                   # This file
```

## 🔧 Configuration

### Frontend Configuration

**Local (Environment Variable):**
```bash
export BACKEND_API_URL="https://your-modal-url.modal.run"
```

**Local (Streamlit secrets):**
Create `app/.streamlit/secrets.toml`:
```toml
BACKEND_API_URL = "https://your-modal-url.modal.run"
```

**Streamlit Cloud:**
Add secret in dashboard with key `BACKEND_API_URL`

### Backend Configuration (Modal)

**Modal Secrets:**
Set via Modal dashboard or CLI:
```bash
modal secret create env \
  API_URL=https://your-api-url.com \
  API_KEY=your_api_key \
  MODEL_NAME=tngtech/deepseek-r1t2-chimera:free
```

**Note**: API configuration is optional. If not provided, the pipeline returns combined frame captions without final API summarization.

### Pipeline Parameters

Customize in `backend/pipeline.py`:
- `frame_skip`: Extract every Nth frame (default: 30)
- `importance_threshold`: Threshold for frame selection (default: 0.5)

### Modal Configuration

Edit `backend/modal_app.py`:
- `gpu`: GPU type ("T4", "A10G", "A100", or `None` for CPU)
- `timeout`: Request timeout in seconds (default: 1800)
- `memory`: Memory allocation in MB (default: 8192)

## 🔌 API Endpoints

### `GET /`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "message": "Video Summary Generator API is running",
  "version": "1.0.0"
}
```

### `GET /health`
Detailed health check.

**Response:**
```json
{
  "status": "healthy",
  "uploads_directory": "/path/to/uploads",
  "uploads_directory_exists": true
}
```

### `POST /process_upload`
Process an uploaded video and generate a summary.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Video file (`.mp4`, `.mkv`, or `.mov`)
- Max file size: 500 MB

**Response:**
```json
{
  "status": "success",
  "run_id": "uuid-here",
  "summary": "Generated summary text...",
  "original_filename": "video.mp4",
  "file_size_bytes": 12345678
}
```

**Error Responses:**
- `400`: Unsupported file format or invalid request
- `413`: File size exceeds limit
- `500`: Processing error

## 🧠 ML Pipeline Details

The pipeline consists of several stages:

1. **Frame Extraction**: Uses OpenCV to extract frames at configurable intervals (every 30th frame by default)
2. **Feature Extraction**: ResNet50 extracts 2048-dimensional feature vectors from each frame
3. **Frame Scoring**: Bidirectional LSTM scores frame importance using temporal context
4. **Frame Selection**: Frames above threshold (0.5) are selected as key frames
5. **Caption Generation**: BLIP model generates natural language captions for key frames
6. **Summarization**: Optional API call for final summary refinement (if API_URL configured)

## 🧹 Cleanup

The application includes automatic cleanup features:

- **Frontend**: Old uploads (24+ hours) are automatically removed on app start
- **Backend**: Temporary processing files are cleaned up after each run
- **Manual Cleanup**: Use the sidebar in the frontend to manually clean uploads

## 🐛 Troubleshooting

### Modal Deployment Issues

**Server not responding:**
```bash
# Check logs
modal app logs video-summary-generator

# Test health endpoint
curl https://your-url.modal.run/health

# Redeploy
modal deploy backend/modal_app.py
```

**Cold start delays:**
- First request after inactivity may take 30-60 seconds (model loading)
- This is normal for serverless deployments

**GPU not available:**
- Edit `backend/modal_app.py` and set `gpu=None` for CPU-only

### Frontend Connection Issues

**Backend not connecting:**
- Verify `BACKEND_API_URL` is set correctly
- Test Modal URL: `curl https://your-url.modal.run/health`
- Check CORS (Modal handles this automatically)


## 📊 Performance Expectations

- **Frame Extraction**: ~1-2 seconds per minute of video
- **Feature Extraction**: ~2-5 seconds per frame (CPU) / ~0.5-1 second (GPU)
- **Caption Generation**: ~1-2 seconds per key frame
- **Total Time**: 
  - CPU: ~5-10 minutes for a 1-minute video
  - GPU (T4): ~2-5 minutes for a 1-minute video

## 📝 License

See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For issues and questions:
- Open an issue on the repository
- Check Modal documentation: [modal.com/docs](https://modal.com/docs)
- Check Streamlit documentation: [docs.streamlit.io](https://docs.streamlit.io)

## 🙏 Acknowledgments

- **ResNet50**: Pre-trained model for feature extraction
- **BLIP**: Salesforce's image captioning model
- **Modal**: Serverless GPU infrastructure
- **Streamlit**: Rapid web app framework

---

**Note**: This project is under active development. The pipeline may take several minutes to process videos depending on length and hardware capabilities. First request on Modal may experience cold start delays while models are loaded.
