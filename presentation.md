# Slide 1: Title Slide

**Title:** Video Summary Generator
**Subtitle:** An AI-Powered Application for Automated Video Summarization
**Presenter:** [Your Name]
**Date:** November 13, 2025

---

# Slide 2: Introduction & Problem Statement

*   **The Challenge:** Videos are a primary source of information, but consuming long-form video content is time-consuming. Manually creating summaries is inefficient and requires significant effort.
*   **The Solution:** An automated system that leverages deep learning to analyze video content and generate a concise, text-based summary.
*   **Goal:** Save time and make video content more accessible and searchable.

---

# Slide 3: Key Features

*   **User-Friendly Interface:** Simple web-based UI for video uploads (`.mp4`, `.mkv`, `.mov`).
*   **Intelligent Summarization:** Automatically identifies and captions the most important moments in a video.
*   **Cloud-Native Architecture:** Built for scalability and performance with a serverless backend.
*   **End-to-End Pipeline:** A complete, automated workflow from video input to text summary output.
*   **Secure & Configurable:** Manages API keys and settings securely.

---

# Slide 4: System Architecture

**(Include the architecture diagram from the README)**

*   **Frontend (Streamlit):**
    *   Handles user interaction, file uploads, and displays results.
    *   Communicates with the backend via HTTP requests.
*   **Backend (FastAPI on Modal):**
    *   A serverless API that receives video files.
    *   Orchestrates the machine learning pipeline.
    *   Leverages GPU support for fast processing.
*   **ML Pipeline (PyTorch):**
    *   The core engine that performs the video analysis and summarization.

---

# Slide 5: Technology Stack

*   **Core Frameworks:**
    *   **Python:** The foundation of the project.
    *   **FastAPI:** For building the high-performance backend API.
    *   **Streamlit:** For creating the interactive web frontend.
*   **Deployment & Infrastructure:**
    *   **Modal:** Serverless platform for deploying the GPU-powered backend.
    *   **Streamlit Cloud:** For hosting the frontend application.
*   **Machine Learning:**
    *   **PyTorch:** The primary deep learning framework.
    *   **Hugging Face Transformers:** For accessing the pre-trained BLIP model.
*   **Computer Vision:**
    *   **OpenCV:** For video processing and frame extraction.

---

# Slide 6: The Machine Learning Pipeline (In-Depth)

1.  **Frame Extraction:** The video is sampled to extract frames at regular intervals (e.g., every 30th frame).
2.  **Feature Extraction:** A **ResNet50** model, pre-trained on ImageNet, converts each frame into a numerical feature vector, capturing its visual content.
3.  **Importance Scoring:** A **Bidirectional LSTM** network analyzes the sequence of frame features to understand the temporal context and assigns an "importance score" to each frame.
4.  **Key Frame Selection:** Frames with an importance score above a set threshold are selected as the most significant moments in the video.
5.  **Caption Generation:** The selected key frames are passed to a **BLIP (Bootstrapping Language-Image Pre-training)** model, which generates a descriptive text caption for each frame.
6.  **Final Summarization:** The generated captions are combined to form the final summary. (Optionally, an external language model can be used to refine this into a more coherent paragraph).

---

# Slide 7: Deployment Workflow

*   **Backend on Modal:**
    1.  The FastAPI application is packaged with its dependencies in `modal_app.py`.
    2.  Deployed with a single command: `modal deploy backend/modal_app.py`.
    3.  Modal automatically provisions a container with GPU access (e.g., T4) and exposes a public API endpoint.
*   **Frontend on Streamlit Cloud:**
    1.  The code is pushed to a GitHub repository.
    2.  Streamlit Cloud is linked to the repository.
    3.  The `BACKEND_API_URL` secret is configured to point to the Modal endpoint.
    4.  The application is deployed and becomes publicly accessible.

---

# Slide 8: Demonstration

*   **(This is where you would show a live demo or screenshots)**
*   **Step 1:** Show the Streamlit interface.
*   **Step 2:** Upload a short video.
*   **Step 3:** Click "Process Video" and explain that the backend is now processing the file.
*   **Step 4:** Display the final generated summary.

---

# Slide 9: Conclusion & Future Work

*   **Summary:** We have successfully built an end-to-end application that automates video summarization using a sophisticated deep learning pipeline and a modern, scalable cloud architecture.
*   **Future Work:**
    *   **Real-time Processing:** Support for live video streams.
    *   **Multi-modal Summarization:** Incorporate audio transcription to create even richer summaries.
    *   **Advanced Summarization Models:** Use more powerful language models (like GPT-4 or others) for the final summary generation step.
    *   **Batch Processing:** Allow users to upload and process multiple videos at once.

---

# Slide 10: Thank You & Q&A

*   **Thank you for your attention!**
*   **Questions?**
