
import csv
import os
import sys
import pandas as pd

# Add the backend directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.pipeline import run_pipeline

def test_pipeline():
    """
    Tests the video summarization pipeline by comparing its output
    with premade summaries.
    """
    input_csv = 'notebooks/videos.csv'
    video_dir = 'dataset/MSRVTT/TrainValVideo'
    output_csv = 'notebooks/test_results.csv'

    # Read the input CSV
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"\nError: Input CSV file not found at {input_csv}")
        return

    # Prepare the output file
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video_id', 'premade_summary', 'pipeline_summary_output'])

    # Process a subset of the videos
    for index, row in df.iterrows():
        video_id = row['video_id']
        premade_summary = row['caption']
        video_path = os.path.join(video_dir, f'{video_id}.mp4')

        if not os.path.exists(video_path):
            print(f"\nWarning: Video file not found for {video_id}, skipping.")
            continue

        print(f"\nProcessing video: {video_id}")
        try:
            pipeline_summary = run_pipeline(video_path)
        except Exception as e:
            pipeline_summary = f"Error during processing: {e}"
            print(f"\nError processing {video_id}: {e}")

        # Write the results to the output CSV
        with open(output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([video_id, premade_summary, pipeline_summary])


    print(f"Testing complete. Results saved to {output_csv}")

if __name__ == "__main__":
    test_pipeline()
