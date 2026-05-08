import subprocess
import os
import sys


def extract_frames(video_path, downsample_factor=2):
    # Ensure the ffmpeg is installed
    if subprocess.call(["which", "ffmpeg"]) != 0:
        print("ffmpeg is not installed. Please install ffmpeg to use this script.")
        sys.exit(1)

    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"The video file {video_path} does not exist.")
        sys.exit(1)

    # Define the output folder based on the video file's path
    base_path = os.path.dirname(video_path)
    output_folder = os.path.join(base_path, "images")

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Construct the ffmpeg command
    output_pattern = os.path.join(output_folder, f"%06d.jpg")
    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        f"scale=iw/{downsample_factor}:ih/{downsample_factor}",
        "-qscale:v",
        "1",  # Set the quality scale for video streams to the highest quality
        "-start_number",
        "0",  # Start numbering from 000000
        output_pattern,
    ]

    # Execute the ffmpeg command
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Frames extracted successfully into {output_folder}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during frame extraction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(
            "Usage: python extract_all_frames.py <video_path> [OPTIONAL: downsample_factor]"
        )
        sys.exit(1)

    video_path = sys.argv[1]
    downsample_factor = sys.argv[2] if len(sys.argv) == 3 else 2
    extract_frames(video_path, downsample_factor)
