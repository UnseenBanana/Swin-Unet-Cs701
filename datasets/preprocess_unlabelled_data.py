import os
import argparse
from glob import glob
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def preprocess_unlabeled_images(
    image_files: list, dst_path: str, use_normalize: bool = True
) -> None:
    """
    Preprocess unlabeled images for testing.

    Args:
        image_files: List of image file paths
        dst_path: Destination path for preprocessed files
        use_normalize: Whether to normalize the images
    """
    os.makedirs(os.path.join(dst_path, "test_vol_h5"), exist_ok=True)

    # Normalization parameters for 8-bit PNG images
    a_min, a_max = 0, 255

    # Group images by case
    cases = {}
    pbar = tqdm(image_files, desc="Processing images")
    for image_file in pbar:
        # Extract case number from the file path
        case_num = image_file.split("/")[-2]  # Adjust this based on your file structure

        # Read and process image
        image_data = np.array(Image.open(image_file))

        # Normalize if requested
        if use_normalize:
            image_data = (image_data - a_min) / (a_max - a_min)

        # Initialize case in dictionary if not exists
        if case_num not in cases:
            cases[case_num] = {"images": []}

        cases[case_num]["images"].append(image_data)

    # Save each case as an H5 file
    pbar = tqdm(cases.items(), desc="Saving H5 files")
    for case_num, data in pbar:
        save_path = os.path.join(dst_path, "test_vol_h5", f"case{case_num}.npy.h5")
        with h5py.File(save_path, "w") as f:
            # Stack images into a single array
            images = np.stack(data["images"], axis=0)
            f["image"] = images
            # Create dummy labels of zeros with the same shape
            f["label"] = np.zeros_like(images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_path",
        type=str,
        required=True,
        help="source directory containing the unlabeled images",
    )
    parser.add_argument(
        "--dst_path",
        type=str,
        required=True,
        help="destination directory for preprocessed data",
    )
    parser.add_argument(
        "--use_normalize",
        action="store_true",
        default=True,
        help="normalize the images",
    )

    args = parser.parse_args()

    # Get all PNG files from the source directory
    image_files = sorted(glob(os.path.join(args.src_path, "**/*.png"), recursive=True))

    if not image_files:
        raise ValueError(f"No PNG files found in {args.src_path}")

    print(f"Found {len(image_files)} images to process")

    # Process the images
    preprocess_unlabeled_images(image_files, args.dst_path, args.use_normalize)
