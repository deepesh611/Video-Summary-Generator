import os
import time
import shutil
import tempfile
import requests

from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from video import (
    extract_frames,
    extract_features,
    select_key_frames,
    caption_frames,
    DEVICE
)
from audio import (
    extract_audio_segment,
    transcribe_audio_segment,
    get_video_duration
)


def create_synchronized_clips(video_path: str, clip_duration: float = 2.0):
    """
    Create synchronized (frame, audio) clips at regular time intervals.
    
    This is the foundation of the multimodal approach: instead of extracting
    frames based on FPS, we extract based on TIME, ensuring audio and visual
    are always aligned.
    
    Args:
        video_path: Path to input video
        clip_duration: Duration of each audio segment in seconds (default: 2.0)
        
    Returns:
        List of dictionaries with 'timestamp', 'frame_path', 'audio_path'
    """
    import cv2
    
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    
    # Create temporary directory for clips
    temp_dir = tempfile.mkdtemp(prefix="multimodal_clips_")
    frames_dir = os.path.join(temp_dir, "frames")
    audio_dir = os.path.join(temp_dir, "audio")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    
    clips = []
    current_time = 0.0
    clip_idx = 0
    
    print(f"Creating synchronized clips (duration: {duration:.1f}s, clip_duration: {clip_duration}s)...")
    
    # Extract clips at regular time intervals
    cap = cv2.VideoCapture(video_path)
    
    while current_time < duration:
        # Extract frame at this timestamp
        frame_num = int(current_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save frame
        frame_path = os.path.join(frames_dir, f"frame_{clip_idx:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        # Extract audio segment [current_time, current_time + clip_duration]
        audio_path = os.path.join(audio_dir, f"audio_{clip_idx:04d}.wav")
        
        try:
            extract_audio_segment(
                video_path,
                audio_path,
                start_time=current_time,
                duration=min(clip_duration, duration - current_time)
            )
        except Exception as e:
            print(f"Warning: Failed to extract audio at {current_time}s: {e}")
            audio_path = None
        
        clips.append({
            'timestamp': current_time,
            'frame_path': frame_path,
            'audio_path': audio_path,
            'clip_idx': clip_idx
        })
        
        current_time += clip_duration
        clip_idx += 1
    
    cap.release()
    
    print(f"Created {len(clips)} synchronized clips")
    return clips, temp_dir


def generate_multimodal_captions(clips: list, whisper_model: str = "base"):
    """
    Generate captions for each multimodal moment combining visual and audio.
    
    Args:
        clips: List of clip dictionaries with frame_path and audio_path
        whisper_model: Whisper model size (tiny, base, small, medium, large)
        
    Returns:
        List of multimodal captions with timestamps
    """
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    from PIL import Image
    import torch
    
    # Load BLIP-2 for visual captioning
    print("Loading BLIP-2 model for visual captioning...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip2_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16
    )
    blip2_model = blip2_model.to(DEVICE)
    blip2_model.eval()
    
    multimodal_captions = []
    
    for clip in clips:
        timestamp = clip['timestamp']
        frame_path = clip['frame_path']
        audio_path = clip['audio_path']
        
        # Get visual caption
        image = Image.open(frame_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            generated_ids = blip2_model.generate(**inputs, max_length=50)
            visual_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # Get audio transcription
        audio_text = ""
        if audio_path and os.path.exists(audio_path):
            try:
                audio_text = transcribe_audio_segment(audio_path, model_size=whisper_model)
            except Exception as e:
                print(f"Warning: Audio transcription failed at {timestamp}s: {e}")
                audio_text = ""
        
        # Combine visual and audio
        if audio_text:
            # Both visual and audio available
            combined_caption = f"{visual_caption}. Audio: \"{audio_text}\""
        else:
            # Only visual available
            combined_caption = visual_caption
        
        multimodal_captions.append({
            'timestamp': timestamp,
            'caption': combined_caption,
            'visual': visual_caption,
            'audio': audio_text
        })
    
    return multimodal_captions


def select_key_moments(multimodal_captions: list, target_count: int = 20):
    """
    Select key moments from multimodal captions.
    
    For now, uses simple uniform sampling. Can be enhanced later with:
    - Diversity-based selection using embeddings
    - Importance scoring based on audio/visual content
    - Scene change detection
    
    Args:
        multimodal_captions: List of caption dictionaries
        target_count: Number of key moments to select
        
    Returns:
        List of selected key moments
    """
    import numpy as np
    
    total_moments = len(multimodal_captions)
    
    if total_moments <= target_count:
        return multimodal_captions
    
    # Uniform sampling for now
    indices = np.linspace(0, total_moments - 1, target_count, dtype=int)
    selected_moments = [multimodal_captions[i] for i in indices]
    
    print(f"Selected {len(selected_moments)} key moments from {total_moments} total moments")
    return selected_moments


def generate_multimodal_summary(key_moments: list) -> str:
    """
    Generate final summary from multimodal key moments.
    
    Args:
        key_moments: List of key moment dictionaries with timestamps and captions
        
    Returns:
        Final multimodal summary text
    """
    # Format timeline
    timeline = []
    for moment in key_moments:
        timeline.append(f"[{moment['timestamp']:.1f}s] {moment['caption']}")
    
    timeline_text = "\n".join(timeline)
    
    # Create a structured prompt for the LLM
    prompt = f"""You are analyzing a video. Below is a timeline of key moments with visual descriptions and audio transcriptions.

TIMELINE:
{timeline_text}

Create a concise summary (not more than 5 sentences) that:
- Describes what happens in the video in a natural narrative flow
- Integrates visual and audio information smoothly
- Does NOT include timestamps (like "at 2 seconds" or "[2.0s]")
- Does NOT quote dialogue directly (paraphrase instead)
- Focuses on the main events, actions, and overall story

Write as if you're describing the video to someone who hasn't seen it.

SUMMARY:"""
    
    # Try to use LLM API for better summarization
    print("Attempting to generate narrative summary with LLM...")
    llm_summary = summarize_with_api(prompt)
    
    if llm_summary:
        # LLM summarization successful
        print("✓ LLM summary generated")
        return f"Video Summary:\n\n{llm_summary}\n\n---\n\nDetailed Timeline:\n{timeline_text}"
    else:
        # Fallback: Use formatted timeline only
        print("⚠ LLM unavailable, using timeline format")
        return f"Video Summary:\n\n{timeline_text}"


def summarize_with_api(prompt: str) -> Optional[str]:
    """
    Send a prompt to LLM API and get a summary.
    
    This is a generic function that can be used for any summarization task.
    The caller provides the complete prompt.
    
    Args:
        prompt: The complete prompt to send to the LLM
        
    Returns:
        LLM-generated summary or None if API not configured
    """
    api_url = os.getenv("API_URL")
    api_key = os.getenv("API_KEY")
    model_name = os.getenv("MODEL_NAME", "tngtech/deepseek-r1t2-chimera:free")
    
    # If API is not configured, return None
    if not api_url or not api_key:
        return None
    
    # Retry configuration
    max_retries = 5
    base_delay = 2  # seconds
    
    for attempt in range(max_retries):
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
                        "content": prompt  # Use the provided prompt directly
                    }]
                },
                timeout=60  # 60 second timeout
            )
            
            # Handle rate limiting with exponential backoff
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    # Calculate exponential backoff delay
                    delay = base_delay * (2 ** attempt)
                    print(f"Rate limit hit. Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    print(f"Rate limit exceeded after {max_retries} attempts. Using original summary.")
                    return video_summary_text
            
            response.raise_for_status()
            result = response.json()
            
            # Extract summary from OpenRouter API response
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            
            # Fallback if response structure is different
            return video_summary_text
                
        except requests.exceptions.RequestException as e:
            # Handle network errors with retry
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"API request failed: {str(e)[:100]}. Retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            else:
                print(f"API summarization failed after {max_retries} attempts, using original summary: {str(e)[:100]}")
                return video_summary_text
        except Exception as e:
            # If API call fails for other reasons, return the original summary
            print(f"API summarization failed, using original summary: {str(e)[:100]}")
            return video_summary_text
    
    # Should never reach here, but just in case
    return video_summary_text



def run_pipeline(
    video_path: str, 
    clip_duration: float = 2.0,
    target_moments: int = 20,
    whisper_model: str = "base"
) -> str:
    """
    Run the complete MULTIMODAL video-to-summary pipeline.
    
    This pipeline processes video and audio TOGETHER at each moment in time,
    creating a truly multimodal understanding of the video.
    
    Pipeline steps:
    1. Create synchronized (frame, audio) clips at regular time intervals
    2. Generate multimodal captions combining visual (BLIP-2) and audio (Whisper)
    3. Select key moments from all multimodal captions
    4. Generate final summary from key moments
    
    Args:
        video_path: Path to the input video file
        clip_duration: Duration of each time-based clip in seconds (default: 2.0)
        target_moments: Number of key moments to select (default: 20)
        whisper_model: Whisper model size - tiny/base/small/medium/large (default: base)
        
    Returns:
        str: The generated multimodal summary text
        
    Raises:
        FileNotFoundError: If video_path does not exist
        Exception: If pipeline processing fails
    """
    # Validate that video file exists
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    temp_dir = None
    
    try:
        print("="*60)
        print("MULTIMODAL VIDEO UNDERSTANDING PIPELINE")
        print("="*60)
        
        # Step 1: Create synchronized (frame, audio) clips
        print(f"\n[1/4] Creating synchronized clips...")
        clips, temp_dir = create_synchronized_clips(video_path, clip_duration)
        print(f"✓ Created {len(clips)} time-based multimodal clips")
        
        # Step 2: Generate multimodal captions
        print(f"\n[2/4] Generating multimodal captions...")
        print(f"  - Visual: BLIP-2")
        print(f"  - Audio: Whisper ({whisper_model})")
        multimodal_captions = generate_multimodal_captions(clips, whisper_model)
        print(f"✓ Generated {len(multimodal_captions)} multimodal captions")
        
        # Step 3: Select key moments
        print(f"\n[3/4] Selecting key moments...")
        key_moments = select_key_moments(multimodal_captions, target_count=target_moments)
        print(f"✓ Selected {len(key_moments)} key moments")
        
        # Step 4: Generate final summary
        print(f"\n[4/4] Generating final summary...")
        final_summary = generate_multimodal_summary(key_moments)
        print(f"✓ Summary generated")
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        
        return final_summary.strip()
        
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\n✓ Cleaned up temporary directory: {temp_dir}")