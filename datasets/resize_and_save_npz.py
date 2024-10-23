import os
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm


def resize_and_save_npz(input_path, output_path, target_size=(224, 224)):
    """
    Resize images in NPZ files to target size
    """
    os.makedirs(output_path, exist_ok=True)

    for file in tqdm(os.listdir(input_path)):
        if file.endswith(".npz"):
            data = np.load(os.path.join(input_path, file))
            image = data["image"]
            label = data["label"]

            # Resize image using bilinear interpolation
            image_pil = Image.fromarray(image)
            image_resized = np.array(image_pil.resize(target_size, Image.BILINEAR))

            # Resize label using nearest neighbor to preserve label values
            label_pil = Image.fromarray(label)
            label_resized = np.array(label_pil.resize(target_size, Image.NEAREST))

            # Save resized data
            output_file = os.path.join(output_path, file)
            np.savez(output_file, image=image_resized, label=label_resized)


def resize_and_save_h5(input_path, output_path, target_size=(224, 224)):
    """
    Resize images in H5 files to target size
    """
    os.makedirs(output_path, exist_ok=True)

    for file in tqdm(os.listdir(input_path)):
        if file.endswith(".npy.h5"):
            with h5py.File(os.path.join(input_path, file), "r") as f:
                image = f["image"][:]
                label = f["label"][:]

                # Resize each slice
                resized_images = []
                resized_labels = []

                for i in range(image.shape[0]):
                    # Resize image using bilinear interpolation
                    img_pil = Image.fromarray(image[i])
                    img_resized = np.array(img_pil.resize(target_size, Image.BILINEAR))
                    resized_images.append(img_resized)

                    # Resize label using nearest neighbor
                    lbl_pil = Image.fromarray(label[i])
                    lbl_resized = np.array(lbl_pil.resize(target_size, Image.NEAREST))
                    resized_labels.append(lbl_resized)

                # Stack resized slices
                resized_images = np.stack(resized_images)
                resized_labels = np.stack(resized_labels)

                # Save resized data
                output_file = os.path.join(output_path, file)
                with h5py.File(output_file, "w") as f_out:
                    f_out.create_dataset("image", data=resized_images)
                    f_out.create_dataset("label", data=resized_labels)


if __name__ == "__main__":
    # Paths
    train_npz_input = "./datasets/cs701/train_npz"
    train_npz_output = "./datasets/cs701_224/train_npz"
    test_h5_input = "./datasets/cs701/test_vol_h5"
    test_h5_output = "./datasets/cs701_224/test_vol_h5"

    # Resize training data
    resize_and_save_npz(train_npz_input, train_npz_output)

    # Resize test data
    resize_and_save_h5(test_h5_input, test_h5_output)
