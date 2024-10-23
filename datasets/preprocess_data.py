import os
import argparse
from glob import glob

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    "--src_path",
    type=str,
    default="./data",
    help="source path for your data",
)
parser.add_argument(
    "--dst_path", type=str, default="./datasets/cs701_test_1", help="root dir for data"
)
parser.add_argument(
    "--use_normalize", action="store_true", default=True, help="use normalize"
)
args = parser.parse_args()


def preprocess_train_image(image_files: str, label_files: str) -> None:
    os.makedirs(f"{args.dst_path}/train_npz", exist_ok=True)

    a_min, a_max = 0, 255  # Assuming 8-bit PNG images
    # b_min, b_max = 0.0, 1.0

    pbar = tqdm(zip(image_files, label_files), total=len(image_files))
    for image_file, label_file in pbar:
        # Extract case number and slice number from the file path
        case_num = image_file.split("/")[-2]
        slice_num = image_file.split("/")[-1].split(".")[0]

        image_data = np.array(Image.open(image_file))
        label_data = np.array(Image.open(label_file))

        if args.use_normalize:
            image_data = (image_data - a_min) / (a_max - a_min)

        save_path = f"{args.dst_path}/train_npz/case{case_num}_slice{slice_num}.npz"
        np.savez(save_path, label=label_data, image=image_data)
    pbar.close()


def preprocess_valid_image(image_files: list, label_files: list) -> None:
    os.makedirs(f"{args.dst_path}/test_vol_h5", exist_ok=True)

    a_min, a_max = 0, 255  # Assuming 8-bit PNG images
    # b_min, b_max = 0.0, 1.0

    cases = {}
    pbar = tqdm(zip(image_files, label_files), total=len(image_files))
    for image_file, label_file in pbar:
        case_num = image_file.split("/")[-2]

        image_data = np.array(Image.open(image_file))
        label_data = np.array(Image.open(label_file))

        if args.use_normalize:
            image_data = (image_data - a_min) / (a_max - a_min)

        if case_num not in cases:
            cases[case_num] = {"images": [], "labels": []}

        cases[case_num]["images"].append(image_data)
        cases[case_num]["labels"].append(label_data)

    for case_num, data in cases.items():
        save_path = f"{args.dst_path}/test_vol_h5/case{case_num}.npy.h5"
        with h5py.File(save_path, "w") as f:
            f["image"] = np.array(data["images"])
            f["label"] = np.array(data["labels"])
    pbar.close()


if __name__ == "__main__":
    image_root = f"{args.src_path}/val_images"
    label_root = f"{args.src_path}/val_labels"

    # String sort
    image_files = sorted(glob(f"{image_root}/**/*.png"))
    label_files = sorted(glob(f"{label_root}/**/*.png"))

    preprocess_train_image(image_files, label_files)
    preprocess_valid_image(image_files, label_files)
