# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
from pathlib import Path

import cv2
import tqdm


def alpha_mask(img_fn, mask_fn):
    img = cv2.imread(str(img_fn))
    msk = cv2.imread(str(mask_fn))
    if img.shape != msk.shape:
        msk = cv2.resize(msk, (img.shape[1], img.shape[0]))

    msk = cv2.cvtColor(msk, cv2.COLOR_RGB2GRAY)

    THRESH = 1  # anything above 0 should work
    _, thresh1 = cv2.threshold(msk, THRESH, 255, cv2.THRESH_BINARY)

    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_RGB2RGBA)
    result[thresh1 == 0] = (0, 0, 0, 0)

    return result, msk


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply masks to corresponding images")
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Path to image directory",
    )
    parser.add_argument(
        "mask_dir",
        type=Path,
        help="Path to mask directory",
    )
    parser.add_argument(
        "masked_image_dir",
        type=Path,
        help="Path to masked image directory",
    )
    parser.add_argument(
        "--mask_fn_format",
        type=str,
        default="%07d.png",
        help="Mask filename format",
    )
    parser.add_argument(
        "--standardized_masks_dir",
        type=Path,
        default=None,
        help="Output standardized masks to this directory",
    )
    args = parser.parse_args()

    # create output directories
    args.masked_image_dir.mkdir(parents=True, exist_ok=True)
    if args.standardized_masks_dir is not None:
        args.standardized_masks_dir.mkdir(parents=True, exist_ok=True)

    # iterate over images
    for img_fn in tqdm.tqdm(sorted(args.image_dir.iterdir())):
        # find corresponding mask
        mask_fn = args.mask_dir / (args.mask_fn_format % int(img_fn.stem))
        assert (
            mask_fn.exists()
        ), f"Mask file {mask_fn} for image {img_fn} does not exist"

        # apply mask
        masked_img, standardized_mask = alpha_mask(img_fn, mask_fn)

        # write masked image
        out_fn = args.masked_image_dir / img_fn.with_suffix(".png").name
        cv2.imwrite(str(out_fn), masked_img)

        # write standardized mask
        if args.standardized_masks_dir is not None:
            standardized_mask_fn = args.standardized_masks_dir / (out_fn.name)
            cv2.imwrite(str(standardized_mask_fn), standardized_mask)

    print(f"Wrote masked images to {args.masked_image_dir}")
    if args.standardized_masks_dir is not None:
        print(f"Wrote standardized masks to {args.standardized_masks_dir}")
