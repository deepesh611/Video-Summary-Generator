"""
Video processing pipeline module.

This module contains the deep learning pipeline for processing videos
and generating summaries. Based on the pipeline from notebooks/notebook1.ipynb.
"""

import os
import json
import glob
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import torch
import numpy as np
import torch.nn as nn
import requests
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load environment variables
load_dotenv()

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FrameImportanceLSTM(nn.Module):
    """
    LSTM model for scoring frame importance.
    
    This model takes frame features and outputs importance scores
    for selecting key frames from the video.
    """
    
    def __init__(self, input_size=2048, hidden_size=256, num_layers=2):
        super(FrameImportanceLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        scores = torch.sigmoid(self.fc(h)).squeeze(-1)
        return scores


def extract_frames(video_path: str, output_dir: str, frame_skip: int = 30) -> List[str]:
    """
    Extract frames from video file.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save extracted frames
        frame_skip: Extract every Nth frame (default: 30)
        
    Returns:
        List of paths to extracted frame images
        
    Raises:
        Exception: If video cannot be opened or processed
    """
    os.makedirs(output_dir, exist_ok=True)
    
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise Exception(f"Failed to open video file: {video_path}")
    
    success, image = vidcap.read()
    count = 0
    frame_files = []
    
    while success:
        if count % frame_skip == 0:
            frame_path = os.path.join(output_dir, f"frame_{count}.jpg")
            cv2.imwrite(frame_path, image)
            frame_files.append(frame_path)
        success, image = vidcap.read()
        count += 1
    
    vidcap.release()
    
    if len(frame_files) == 0:
        raise Exception("No frames were extracted from the video")
    
    return sorted(frame_files)


def extract_features(frame_files: List[str], device: torch.device) -> np.ndarray:
    """
    Extract feature vectors from frames using ResNet50.
    
    Args:
        frame_files: List of paths to frame images
        device: Torch device (CPU or CUDA)
        
    Returns:
        numpy array of feature vectors
    """
    # Load pretrained ResNet (remove last fully-connected layer)
    resnet = models.resnet50(weights=True)
    resnet.eval()
    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor = feature_extractor.to(device)

    # Image preprocessing pipeline for ResNet
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    features = []
    
    for path in frame_files:
        image = Image.open(path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = feature_extractor(input_tensor)
            vector = output.squeeze().cpu().numpy().flatten()

        features.append(vector)

    return np.array(features)


def select_key_frames(
    frame_files: List[str], 
    features: np.ndarray, 
    threshold: float = 0.5,
    device: torch.device = None
) -> List[str]:
    """
    Select key frames based on importance scores from LSTM model.
    
    Args:
        frame_files: List of paths to all extracted frames
        features: Feature vectors for each frame
        threshold: Threshold for selecting frames (default: 0.5)
        device: Torch device (CPU or CUDA)
        
    Returns:
        List of paths to selected key frames
    """
    if device is None:
        device = DEVICE
    
    # Instantiate and load the LSTM model
    model = FrameImportanceLSTM()
    model.eval()
    model = model.to(device)

    # Convert features to tensor and score frames
    features_tensor = torch.tensor(features).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        importance_scores = model(features_tensor).squeeze().cpu().numpy()

    # Select key frames above threshold
    key_indices = np.where(importance_scores > threshold)[0]
    key_frames = [frame_files[i] for i in key_indices]
    
    # If no frames pass threshold, select top frames
    if len(key_frames) == 0:
        top_k = min(10, len(frame_files))
        top_indices = np.argsort(importance_scores)[-top_k:][::-1]
        key_frames = [frame_files[i] for i in top_indices]
    
    return key_frames


def caption_frames(
    frame_files: List[str], 
    device: torch.device = None
) -> List[str]:
    """
    Generate captions for frames using BLIP model.
    
    Args:
        frame_files: List of paths to frame images
        device: Torch device (CPU or CUDA)
        
    Returns:
        List of captions for each frame
    """
    if device is None:
        device = DEVICE
    
    # Load BLIP model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    caption_model = caption_model.to(device)
    caption_model.eval()
    
    def caption_frame(image_path: str) -> str:
        """Generate caption for a single frame."""
        raw_image = Image.open(image_path).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = caption_model.generate(**inputs, max_length=50)
            caption = processor.decode(out[0], skip_special_tokens=True)
        
        return caption
    
    # Generate captions for all key frames
    video_summary = []
    for idx, frame_path in enumerate(frame_files):
        caption = caption_frame(frame_path)
        video_summary.append(f"Scene {idx+1}: {caption}")
    
    return video_summary


def summarize_with_api(video_summary_text: str) -> Optional[str]:
    """
    Send video summary to external API for final summarization.
    
    Args:
        video_summary_text: Combined text from all frame captions
        
    Returns:
        Final summary text from API, or None if API is not configured
    """
    api_url = os.getenv("API_URL")
    api_key = os.getenv("API_KEY")
    model_name = os.getenv("MODEL_NAME", "tngtech/deepseek-r1t2-chimera:free")
    
    # If API is not configured, return None
    if not api_url or not api_key:
        return None
    
    try:
        response = requests.post(
            api_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model_name,
                "messages": [{
                    "role": "user",
                    "content": f'''Rewrite and summarize this video breakdown in as much detail as possible for a general audience. Only give the summary, no other intro or outro.:

{video_summary_text}'''
                }]
            },
            timeout=60  # 60 second timeout
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Extract summary from API response
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            # Fallback if response structure is different
            return video_summary_text
            
    except Exception as e:
        # If API call fails, return the original summary
        # Don't expose API errors to avoid leaking configuration
        print(f"API summarization failed, using original summary: {str(e)[:100]}")
        return video_summary_text


def run_pipeline(video_path: str, frame_skip: int = 30, importance_threshold: float = 0.5) -> str:
    """
    Run the complete video-to-summary pipeline.
    
    Pipeline steps:
    1. Extract frames from video
    2. Extract features using ResNet50
    3. Score frames using LSTM model
    4. Select key frames
    5. Generate captions for key frames using BLIP
    6. Optionally summarize using external API
    
    Args:
        video_path: Path to the input video file
        frame_skip: Extract every Nth frame (default: 30)
        importance_threshold: Threshold for frame selection (default: 0.5)
        
    Returns:
        str: The generated summary text
        
    Raises:
        FileNotFoundError: If video_path does not exist
        Exception: If pipeline processing fails
    """
    # Validate that video file exists
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp(prefix="video_pipeline_")
    frames_dir = os.path.join(temp_dir, "frames")
    
    try:
        # Step 1: Extract frames from video
        print(f"Extracting frames from video...")
        frame_files = extract_frames(video_path, frames_dir, frame_skip)
        print(f"Extracted {len(frame_files)} frames")
        
        # Step 2: Extract features using ResNet50
        print(f"Extracting features using ResNet50...")
        features = extract_features(frame_files, DEVICE)
        print(f"Extracted features with shape: {features.shape}")
        
        # Step 3: Select key frames using LSTM model
        print(f"Selecting key frames...")
        key_frames = select_key_frames(
            frame_files, 
            features, 
            threshold=importance_threshold,
            device=DEVICE
        )
        print(f"Selected {len(key_frames)} key frames")
        
        # Step 4: Generate captions for key frames using BLIP
        print(f"Generating captions for key frames...")
        video_summary_list = caption_frames(key_frames, DEVICE)
        
        # Combine captions into text
        video_summary_text = "\n".join(video_summary_list)
        
        # Step 5: Optionally summarize using external API
        print(f"Summarizing with API...")
        final_summary = summarize_with_api(video_summary_text)
        
        # If API summarization failed or is not configured, use combined captions
        if final_summary is None:
            # Fallback: Use a simple combination of captions
            final_summary = "Video Summary:\n\n" + video_summary_text
        
        return final_summary.strip()
        
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")