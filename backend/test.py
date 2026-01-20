from pipeline import run_pipeline

# Test with a short video
summary = run_pipeline(
    "path/to/test_video.mp4",
    clip_duration=2.0,      # 2-second clips
    target_moments=10,      # 10 key moments
    whisper_model="tiny"    # Fast model for testing
)
print(summary)