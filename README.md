# Video Summary Generator

A deep learning-based application that generates text summaries from video inputs. The project consists of a Streamlit frontend for video upload and a FastAPI backend that processes videos through a sophisticated ML pipeline to generate summaries.

## 🎬 Features

- **Video Upload Interface**: User-friendly Streamlit frontend supporting `.mp4`, `.mkv`, and `.mov` formats
- **Intelligent Frame Extraction**: Automatically extracts and selects key frames from videos
- **Deep Learning Pipeline**:
  - Frame feature extraction using ResNet50
  - Frame importance scoring with LSTM model
  - Key frame selection based on importance thresholds
  - Caption generation using BLIP (Salesforce/blip-image-captioning-base)
  - Optional API-based final summarization
- **Automatic Cleanup**: Temporary files and old uploads are automatically managed
- **RESTful API**: FastAPI backend with proper error handling and CORS support
- **Secure Configuration**: Environment-based configuration for API keys and secrets

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended for faster processing)
- Minimum 4GB RAM (8GB+ recommended)
- Disk space for video storage and model cache

## 🏗️ Architecture

The project is split into two main components:

### Frontend (`app/`)
- **Streamlit Application**: Web-based UI for video upload and result display
- Handles file upload, displays processing status, and manages user sessions
- Automatic cleanup of old uploads (24+ hours)

### Backend (`backend/`)
- **FastAPI Server**: RESTful API for video processing
- **ML Pipeline**: Complete deep learning workflow for video-to-text summarization
  - Video frame extraction (OpenCV)
  - Feature extraction (ResNet50)
  - Frame importance scoring (LSTM)
  - Caption generation (BLIP)
  - Optional API summarization

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Video-Summary-Generator
```

### 2. Create Virtual Environment

```bash
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies

**Option A: Install from root requirements.txt (includes all dependencies)**
```bash
pip install -r requirements.txt
```

**Option B: Install separately for frontend and backend**

Frontend:
```bash
cd app
pip install -r requirements.txt
```

Backend:
```bash
cd backend
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create `backend/.env` file:

```bash
cd backend
cp .env.example .env  # If .env.example exists, or create manually
```

Edit `backend/.env`:
```env
API_URL=https://your-api-url.com
API_KEY=your_api_key_here
MODEL_NAME=tngtech/deepseek-r1t2-chimera:free
```

**Note**: The API configuration is optional. If not provided, the pipeline will return combined frame captions without final API summarization.

## 📖 Usage

### Running the Frontend

```bash
cd app
streamlit run main.py
```

The frontend will be available at `http://localhost:8501`

### Running the Backend

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

**API Documentation**: Visit `http://localhost:8000/docs` for interactive API documentation.

### Using the Application

1. **Upload Video**: Use the Streamlit frontend to upload a video file
2. **Processing**: The backend will:
   - Extract frames from the video
   - Extract features and score frame importance
   - Generate captions for key frames
   - Optionally summarize using external API
3. **View Results**: The summary will be displayed in the frontend

### API Usage (Direct)

You can also call the API directly:

```bash
curl -X POST "http://localhost:8000/process_upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_video.mp4"
```

Response:
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
│   ├── requirements.txt        # Frontend dependencies
│   ├── uploads/                # User uploads (auto-created, gitignored)
│   └── .streamlit/             # Streamlit configuration
│       └── secrets.toml        # Secrets (gitignored)
│
├── backend/                     # FastAPI Backend
│   ├── main.py                 # FastAPI application
│   ├── pipeline.py             # ML pipeline implementation
│   ├── requirements.txt        # Backend dependencies
│   ├── uploads/                # Processed videos (auto-created, gitignored)
│   └── .env                    # Environment variables (gitignored)
│
├── notebooks/                   # Development notebooks
│   └── notebook1.ipynb         # Original pipeline development
│
├── requirements.txt            # Root-level dependencies (all packages)
├── setup.sh                    # Setup script
├── .gitignore                  # Git ignore rules
├── LICENSE                     # License file
└── README.md                   # This file
```

## 🔧 Configuration

### Frontend Configuration

The frontend uses Streamlit's standard configuration. Secrets can be stored in `app/.streamlit/secrets.toml`:

```toml
[secrets]
# Add any frontend secrets here
```

### Backend Configuration

Backend configuration is done via `backend/.env`:

```env
# Required for API summarization (optional)
API_URL=https://api.example.com/v1/summarize
API_KEY=your_api_key_here
MODEL_NAME=tngtech/deepseek-r1t2-chimera:free

# Optional: Processing settings
MAX_VIDEO_DURATION_SECONDS=600
FRAME_EXTRACTION_RATE=1
```

### Pipeline Parameters

The pipeline can be customized in `backend/pipeline.py`:

- `frame_skip`: Extract every Nth frame (default: 30)
- `importance_threshold`: Threshold for frame selection (default: 0.5)

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

1. **Frame Extraction**: Uses OpenCV to extract frames at configurable intervals
2. **Feature Extraction**: ResNet50 extracts 2048-dimensional feature vectors
3. **Frame Scoring**: Bidirectional LSTM scores frame importance
4. **Frame Selection**: Frames above threshold are selected as key frames
5. **Caption Generation**: BLIP model generates natural language captions
6. **Summarization**: Optional API call for final summary refinement

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Backend**: FastAPI, Uvicorn
- **Deep Learning**: PyTorch, Torchvision, Transformers
- **Computer Vision**: OpenCV, Pillow
- **Video Processing**: OpenCV
- **NLP**: Transformers (BLIP, optional T5)
- **Utilities**: python-dotenv, requests

## 🔐 Security Notes

- API keys and secrets are loaded from `.env` files (gitignored)
- Sensitive information is filtered from error messages
- Upload directories are automatically cleaned up
- File size limits prevent resource exhaustion

## 🧹 Cleanup

The application includes automatic cleanup features:

- **Frontend**: Old uploads (24+ hours) are automatically removed on app start
- **Backend**: Temporary processing files are cleaned up after each run
- **Manual Cleanup**: Use the sidebar in the frontend to manually clean uploads

## 🐛 Troubleshooting

### CUDA/GPU Issues
- If CUDA is not available, the pipeline will run on CPU (slower)
- Check GPU availability: `torch.cuda.is_available()`

### Model Download
- Models are automatically downloaded on first run (BLIP, ResNet50)
- Ensure stable internet connection for initial setup
- Models are cached in `~/.cache/huggingface/` and `~/.cache/torch/`

### Memory Issues
- Reduce `frame_skip` to process fewer frames
- Use smaller videos for testing
- Consider increasing system RAM or using cloud instances

### API Errors
- Check `backend/.env` configuration
- Verify API credentials are correct
- If API fails, pipeline falls back to combined captions

## 📝 License

See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🚧 Future Improvements

- [ ] Batch processing endpoint
- [ ] Real-time processing status updates
- [ ] Support for more video formats
- [ ] Trained LSTM model checkpoint loading
- [ ] Caching for processed videos
- [ ] User authentication
- [ ] Progress bars and status tracking
- [ ] Multiple summarization strategies
- [ ] Export summaries in multiple formats

## 📞 Support

For issues and questions, please open an issue on the repository.

---

**Note**: This project is under active development. The pipeline may take several minutes to process videos depending on length and hardware capabilities.