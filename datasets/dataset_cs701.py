import os
import random

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # Random augmentation step
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        # Resize step
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(
                image, (self.output_size[0] / x, self.output_size[1] / y), order=3
            )  # why not 3?
            label = zoom(
                label, (self.output_size[0] / x, self.output_size[1] / y), order=0
            )

        # Conversion to tensor step
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {"image": image, "label": label.long()}
        return sample


class Cs701_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split

        # Read list of samples from the appropriate split file (train.txt, val.txt, etc.)
        self.sample_list = open(os.path.join(list_dir, self.split + ".txt")).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # Handle .npz files (training data)
        if self.split in ["train", "val"] or self.sample_list[idx].strip("\n").split(
            ","
        )[0].endswith(".npz"):
            # Extract filename from the sample list
            slice_name = self.sample_list[idx].strip("\n").split(",")[0]

            # Construct full file path
            if slice_name.endswith(".npz"):
                data_path = os.path.join(self.data_dir, slice_name)
            else:
                data_path = os.path.join(self.data_dir, slice_name + ".npz")

            data = np.load(data_path)

            # Try (image / label) for image and label,
            # fall back to (data / seg) if it fails
            try:
                image, label = data["image"], data["label"]
            except Exception:
                image, label = data["data"], data["seg"]

        # Handle .h5 files (test data)
        else:
            # Get volume name and construct filepath
            vol_name = self.sample_list[idx].strip("\n")
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)

            # Load data from H5 file
            data = h5py.File(filepath)
            image, label = data["image"][:], data["label"][:]

        # Create sample dictionary with image and label
        sample = {"image": image, "label": label}

        # Apply transforms if specified
        if self.transform:
            sample = self.transform(sample)

        # Add case name to the sample dictionary for reference
        sample["case_name"] = self.sample_list[idx].strip("\n")

        return sample
