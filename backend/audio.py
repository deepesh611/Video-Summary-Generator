import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional


def extract_audio(
    video_path: str, 
    output_path: str, 
    sample_rate: int = 16000
) -> str:
    """
    Extract complete audio track from video file using FFmpeg.
    
    Args:
        video_path: Path to input video file
        output_path: Path for output audio file (.wav)
        sample_rate: Audio sample rate in Hz (default: 16000 for Whisper)
    
    Returns:
        Path to extracted audio file
        
    Raises:
        Exception: If FFmpeg extraction fails
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # WAV codec
        '-ar', str(sample_rate),  # Sample rate
        '-ac', '1',  # Mono
        '-y',  # Overwrite output file
        output_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        return output_path
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to extract audio: {e.stderr}")


def extract_audio_segment(
    video_path: str,
    output_path: str,
    start_time: float,
    duration: float,
    sample_rate: int = 16000
) -> str:
    """
    Extract audio segment from video at specific timestamp.
    
    This is used to create synchronized (frame, audio) pairs where
    each audio segment corresponds to a specific time window.
    
    Args:
        video_path: Path to input video file
        output_path: Path for output audio segment (.wav)
        start_time: Start time in seconds
        duration: Duration of segment in seconds
        sample_rate: Audio sample rate in Hz (default: 16000)
    
    Returns:
        Path to extracted audio segment
        
    Raises:
        Exception: If FFmpeg extraction fails
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-ss', str(start_time),
        '-t', str(duration),
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # WAV codec
        '-ar', str(sample_rate),  # Sample rate
        '-ac', '1',  # Mono
        '-y',  # Overwrite output file
        output_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        return output_path
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to extract audio segment: {e.stderr}")


def transcribe_audio(
    audio_path: str, 
    model_size: str = "base",
    language: Optional[str] = None
) -> Dict:
    """
    Transcribe audio to text using OpenAI Whisper.
    
    Args:
        audio_path: Path to audio file
        model_size: Whisper model size - one of:
            - 'tiny': Fastest, least accurate (~1GB RAM)
            - 'base': Good balance (~1GB RAM)
            - 'small': Better accuracy (~2GB RAM)
            - 'medium': High accuracy (~5GB RAM)
            - 'large': Best accuracy (~10GB RAM)
        language: Language code (e.g., 'en', 'es', 'fr') or None for auto-detect
    
    Returns:
        Dictionary containing:
            - 'text': Full transcription text
            - 'segments': List of segments with timestamps
            - 'language': Detected or specified language
            
    Example:
        result = transcribe_audio('audio.wav', model_size='base')
        print(result['text'])  # "Hello, this is a test."
        print(result['segments'][0])  # {'start': 0.0, 'end': 2.5, 'text': '...'}
    """
    import whisper
    
    # Load Whisper model (cached after first load)
    print(f"Loading Whisper {model_size} model...")
    model = whisper.load_model(model_size)
    
    # Transcribe with word-level timestamps
    print(f"Transcribing audio: {audio_path}")
    result = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,  # Get word-level timing
        verbose=False  # Suppress progress output
    )
    
    return {
        'text': result['text'].strip(),
        'segments': result['segments'],
        'language': result['language']
    }


def transcribe_audio_segment(
    audio_path: str,
    model_size: str = "base"
) -> str:
    """
    Transcribe a short audio segment and return just the text.
    
    Simplified version of transcribe_audio() for quick transcription
    of short segments (e.g., 2-second clips).
    
    Args:
        audio_path: Path to audio segment
        model_size: Whisper model size (default: 'base')
    
    Returns:
        Transcribed text string
    """
    result = transcribe_audio(audio_path, model_size=model_size)
    return result['text']


def get_audio_duration(audio_path: str) -> float:
    """
    Get duration of audio file in seconds using FFmpeg.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Duration in seconds
        
    Raises:
        Exception: If FFmpeg probe fails
    """
    cmd = [
        'ffprobe',
        '-i', audio_path,
        '-show_entries', 'format=duration',
        '-v', 'quiet',
        '-of', 'csv=p=0'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        duration = float(result.stdout.strip())
        return duration
    except (subprocess.CalledProcessError, ValueError) as e:
        raise Exception(f"Failed to get audio duration: {e}")


def get_video_duration(video_path: str) -> float:
    """
    Get duration of video file in seconds using FFmpeg.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Duration in seconds
    """
    return get_audio_duration(video_path)  # Same FFprobe command works for video


def detect_audio_events(audio_path: str, threshold: float = 0.3) -> List[Dict]:
    """
    Detect audio events (music, speech, sounds) using PANNs.
    
    NOTE: This is an optional advanced feature. Requires panns-inference package.
    Can be implemented later for richer audio understanding.
    
    Args:
        audio_path: Path to audio file
        threshold: Confidence threshold for event detection
    
    Returns:
        List of detected events with labels and confidence scores
        
    Example:
        events = detect_audio_events('audio.wav')
        # [{'label': 'Speech', 'confidence': 0.95}, 
        #  {'label': 'Music', 'confidence': 0.72}]
    """
    raise NotImplementedError(
        "Audio event detection not yet implemented. "
        "This is an optional feature for future enhancement."
    )


# Helper function for creating synchronized clips
def create_audio_segments_for_timestamps(
    video_path: str,
    timestamps: List[float],
    segment_duration: float = 2.0,
    output_dir: Optional[str] = None
) -> List[Dict]:
    """
    Create audio segments for a list of timestamps.
    
    This is a convenience function for batch processing multiple timestamps.
    
    Args:
        video_path: Path to input video
        timestamps: List of timestamps in seconds
        segment_duration: Duration of each segment in seconds
        output_dir: Output directory (creates temp dir if None)
    
    Returns:
        List of dictionaries with 'timestamp' and 'audio_path'
        
    Example:
        segments = create_audio_segments_for_timestamps(
            'video.mp4',
            timestamps=[0.0, 2.0, 4.0],
            segment_duration=2.0
        )
        # [{'timestamp': 0.0, 'audio_path': '/tmp/audio_0.wav'}, ...]
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="audio_segments_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    segments = []
    
    for idx, timestamp in enumerate(timestamps):
        output_path = os.path.join(output_dir, f"audio_{idx:04d}.wav")
        
        try:
            extract_audio_segment(
                video_path,
                output_path,
                start_time=timestamp,
                duration=segment_duration
            )
            
            segments.append({
                'timestamp': timestamp,
                'audio_path': output_path,
                'index': idx
            })
        except Exception as e:
            print(f"Warning: Failed to extract audio at {timestamp}s: {e}")
            continue
    
    return segments
