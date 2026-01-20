import cv2
import torch
import warnings
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from dotenv import load_dotenv
from typing import List, Optional


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
    
    def __init__(self, input_size=1024, hidden_size=256, num_layers=2):
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


def extract_frames(
    video_path: str, 
    output_dir: str, 
    frame_skip: int = 30
    ) -> List[str]:
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


def extract_features(
    frame_files: List[str], 
    device: torch.device
    ) -> np.ndarray:
    # Load pretrained GoogLeNet and disable auxiliary classifiers
    try:
        weights = models.GoogLeNet_Weights.DEFAULT
        googlenet = models.googlenet(weights=weights, aux_logits=True)
    except Exception:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            googlenet = models.googlenet(pretrained=True, aux_logits=True)

    googlenet.eval()
    googlenet = googlenet.to(device)

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
            x = input_tensor
            for name, module in googlenet.named_children():
                if name in ('aux1', 'aux2'):
                    continue
                if name == 'fc':
                    break
                x = module(x)

            if x.dim() > 2:
                vector = torch.flatten(x, 1).cpu().numpy().flatten()
            else:
                vector = x.cpu().numpy().flatten()

        features.append(vector)

    return np.array(features)


def select_key_frames(
    frame_files: List[str], 
    features: np.ndarray, 
    threshold: float = 0.5,
    device: torch.device = None
) -> List[str]:
    """
    Select key frames using a pre-trained video model (R(2+1)D).
    Falls back to uniform sampling if model fails.
    
    Args:
        frame_files: List of paths to frame images
        features: (Unused) Feature vectors
        threshold: (Unused) Threshold
        device: Torch device
        
    Returns:
        List of selected key frame paths
    """
    if device is None:
        device = DEVICE
    
    num_frames = len(frame_files)
    target_count = 25  # Aim for 25 frames
    
    # Fallback to uniform sampling if too few frames
    if num_frames <= target_count:
        return frame_files
    
    try:
        # Load pre-trained R(2+1)D model
        from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
        
        weights = R2Plus1D_18_Weights.DEFAULT
        model = r2plus1d_18(weights=weights)
        model = model.to(device)
        model.eval()
        
        # Preprocessing for R(2+1)D (expects 3x16xHxW)
        preprocess = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                               std=[0.22803, 0.22145, 0.216989])
        ])
        
        # Score frames using temporal clips
        clip_size = 16
        scores = []
        
        for i in range(0, num_frames, clip_size // 2):  # 50% overlap
            clip_frames = frame_files[i:i+clip_size]
            
            # Pad if needed
            while len(clip_frames) < clip_size:
                clip_frames.append(clip_frames[-1])
            
            # Load and preprocess frames
            clip_tensors = []
            for frame_path in clip_frames:
                img = Image.open(frame_path).convert('RGB')
                clip_tensors.append(preprocess(img))
            
            # Stack to (C, T, H, W) format
            clip_tensor = torch.stack(clip_tensors, dim=1).unsqueeze(0).to(device)
            
            # Get prediction confidence
            with torch.no_grad():
                output = model(clip_tensor)
                confidence = torch.softmax(output, dim=1).max().item()
            
            # Assign score to middle frame of clip
            mid_idx = i + clip_size // 2
            if mid_idx < num_frames:
                scores.append((mid_idx, confidence))
        
        # Sort by score and select top frames
        scores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = sorted([idx for idx, _ in scores[:target_count]])
        
        return [frame_files[i] for i in selected_indices if i < num_frames]
        
    except Exception as e:
        print(f"R(2+1)D model failed: {e}. Falling back to uniform sampling.")
        # Fallback to uniform sampling
        indices = np.linspace(0, num_frames - 1, target_count, dtype=int)
        return [frame_files[i] for i in indices]


def caption_frames(
    frame_files: List[str], 
    device: torch.device = None
) -> List[str]:
    """
    Generate captions for frames using BLIP-2 model.
    
    Args:
        frame_files: List of paths to frame images
        device: Torch device (CPU or CUDA)
        
    Returns:
        List of captions for each frame
    """
    if device is None:
        device = DEVICE
    
    # Load BLIP-2 model and processor (better accuracy than BLIP)
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    
    print("Loading BLIP-2 model...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    caption_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        dtype=torch.float16
    )
    caption_model = caption_model.to(device)
    caption_model.eval()
    
    def caption_frame(image_path: str) -> str:
        """Generate caption for a single frame."""
        raw_image = Image.open(image_path).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = caption_model.generate(**inputs, max_length=50)
            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        return caption
    
    # Generate captions for all key frames
    video_summary = []
    for idx, frame_path in enumerate(frame_files):
        caption = caption_frame(frame_path)
        video_summary.append(f"Scene {idx+1}: {caption}")
    
    return video_summary