# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
from math import ceil
from pathlib import Path

import cv2
import tqdm
from decord import VideoReader


# helper functions
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def image_sharpness(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    largest_dim = max(image.shape[:2])
    scale_factor = 640 / largest_dim
    image = cv2.resize(image, dsize=(0, 0), fx=scale_factor, fy=scale_factor)
    sharpness = variance_of_laplacian(image)
    return sharpness


# args
parser = argparse.ArgumentParser(
    description="Extract the sharpest frames from a video."
)
parser.add_argument(
    "video_fn",
    type=Path,
    help="the video to load",
)
parser.add_argument(
    "images_dir",
    type=Path,
    help="directory for extracted video frames",
)
parser.add_argument(
    "--min_sharpness_window_size",
    default=10,
    type=int,
    help="minimum window size for sharpness filter (default=10)",
)
parser.add_argument(
    "--max_frame_count",
    default=125,
    type=int,
    help="maximum number of frames to extract (default=125)",
)
parser.add_argument(
    "--sharpness_threshold",
    default=-1,
    type=float,
    help="threshold to filter out blurry images (default=-1)",
)
parser.add_argument(
    "--image_ext",
    default="jpg",
    choices=["jpg", "png"],
    type=str,
    help="frame image extension (default=jpg)",
)
args = parser.parse_args()

# ensure files/paths exist
assert args.video_fn.exists(), f"Video file {args.video_fn} does not exist."
args.images_dir.mkdir(exist_ok=True)

# create video reader
with open(args.video_fn, "rb") as fp:
    video_reader = VideoReader(fp)
assert len(video_reader) > 0, f"Video file {args.video_fn} is empty."

# compute sharpness filter window size
if (
    args.min_sharpness_window_size <= 0
    or len(video_reader) / args.min_sharpness_window_size > args.max_frame_count
):
    sharpness_window_size = ceil(len(video_reader) / args.max_frame_count)
else:
    sharpness_window_size = args.min_sharpness_window_size

print(
    f"Extracting frames from {args.video_fn} with sharpness window {sharpness_window_size} ..."
)
frame_queue, n_frames_written = [], 0
for idx, image in enumerate(tqdm.tqdm(video_reader)):
    # package frame
    image = image.asnumpy()
    frame = {
        "idx": idx,
        "image": image,
        "sharpness": image_sharpness(image),
    }
    frame_queue.append(frame)

    # keep sharpest image in window
    if len(frame_queue) >= sharpness_window_size:
        # get sharpest frame
        frames_by_sharpness = [(frame["sharpness"], frame) for frame in frame_queue]
        sharpeness_val, sharpest = max(frames_by_sharpness)
        frame_queue = []

        # save frame
        if sharpeness_val > args.sharpness_threshold:
            idx, image, sharpness = sharpest.values()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_fn = args.images_dir / f"{idx:06d}.{args.image_ext}"
            n_frames_written += 1

            if args.image_ext == "jpg":
                cv2.imwrite(str(image_fn), image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            elif args.image_ext == "png":
                cv2.imwrite(str(image_fn), image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            else:
                raise ValueError(f"Unsupported image extension {args.image_ext}.")

print(f"{n_frames_written} frames written to {args.images_dir}")
