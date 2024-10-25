import numpy as np
import os
from PIL import Image
import argparse
import nibabel as nib


def convert_nifti_to_png(input_dir, output_dir, target_size=(224, 224)):
    """
    Convert NIfTI files (.nii.gz) to PNG format while preserving the labels.
    Creates a separate folder for each .nii.gz file containing all slices.

    Args:
        input_dir: Directory containing .nii.gz files
        output_dir: Directory to save PNG files
        target_size: Tuple of (width, height) for resizing
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Color mapping for visualization
    colors = [
        (0, 0, 0),  # 0: Background - Black
        (255, 0, 0),  # 1: Red
        (0, 255, 0),  # 2: Green
        (0, 0, 255),  # 3: Blue
        (255, 255, 0),  # 4: Yellow
        (255, 0, 255),  # 5: Magenta
        (0, 255, 255),  # 6: Cyan
        (128, 0, 0),  # 7: Maroon
        (0, 128, 0),  # 8: Dark Green
        (0, 0, 128),  # 9: Navy
        (128, 128, 0),  # 10: Olive
        (128, 0, 128),  # 11: Purple
        (0, 128, 128),  # 12: Teal
    ]

    # Process each NIfTI file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith("pred.nii.gz"):  # Process only prediction files
            # Create a subfolder for this nifti file
            base_name = os.path.splitext(os.path.splitext(filename)[0])[0]
            nifti_output_dir = os.path.join(output_dir, base_name)
            os.makedirs(nifti_output_dir, exist_ok=True)

            # Create separate folders for colored and label images
            colored_output_dir = os.path.join(nifti_output_dir, "colored")
            labels_output_dir = os.path.join(nifti_output_dir, "labels")
            os.makedirs(colored_output_dir, exist_ok=True)
            os.makedirs(labels_output_dir, exist_ok=True)

            # Load NIfTI file
            nifti_path = os.path.join(input_dir, filename)
            nifti_img = nib.load(nifti_path)

            # Get the data array
            data = nifti_img.get_fdata()

            # Process each slice
            num_slices = data.shape[2]

            print(f"Processing {filename} - {num_slices} slices")

            for slice_idx in range(num_slices):
                # Get current slice
                slice_data = data[:, :, slice_idx]

                # Normalize data to 0-12 range if necessary
                if slice_data.max() > 12:
                    slice_data = (slice_data / slice_data.max() * 12).astype(np.uint8)
                else:
                    slice_data = slice_data.astype(np.uint8)

                # Create colored image
                height, width = slice_data.shape
                colored_image = np.zeros((height, width, 3), dtype=np.uint8)

                # Apply colors based on labels
                for label in range(13):
                    mask = slice_data == label
                    colored_image[mask] = colors[label]

                # Convert numpy array to PIL Image
                colored_pil = Image.fromarray(colored_image)
                label_pil = Image.fromarray(slice_data)

                # Resize if necessary
                if (height, width) != target_size:
                    colored_pil = resize_png(colored_pil, target_size)
                    label_pil = resize_png(label_pil, target_size)

                # Save as PNG with slice number in filename
                colored_output_path = os.path.join(
                    colored_output_dir, f"slice_{slice_idx:04d}.png"
                )
                label_output_path = os.path.join(
                    labels_output_dir, f"slice_{slice_idx:04d}.png"
                )

                # Save both colored and label versions
                colored_pil.save(colored_output_path)
                label_pil.save(label_output_path)

                if slice_idx % 10 == 0:  # Progress update every 10 slices
                    print(f"  Processed slice {slice_idx}/{num_slices}")

            print(f"Completed processing {filename}")


def resize_png(input_image, target_size=(512, 512)):
    """
    Resize a PIL Image while preserving its characteristics (labels or RGB).

    Args:
        input_image: Input PIL Image
        target_size: Tuple of (width, height) for resizing

    Returns:
        Resized PIL Image
    """
    # For label images, use NEAREST to preserve label values
    # For RGB images, use BILINEAR for smoother results
    resample_method = Image.NEAREST if input_image.mode == "L" else Image.BILINEAR

    return input_image.resize(target_size, resample=resample_method)


def organise_dir():
    # Organize the directory structure
    old_dir = "./model_out/predictions_png"
    new_dir = "./datasets/data/test_labels"
    os.makedirs(new_dir, exist_ok=True)

    # Copy the files to the new directory
    for folder in os.listdir(old_dir):
        # extract case number and create a directory for it
        case_num = folder.split("_")[0][4:]
        case_dir = os.path.join(new_dir, case_num)
        os.makedirs(case_dir, exist_ok=True)

        # copy the files to the new directory
        for file in os.listdir(os.path.join(old_dir, folder, "labels")):
            old_file = os.path.join(old_dir, folder, "labels", file)

            # rename the file to remove the slice
            slice = file.split("_")[1]
            slice_number = slice.split(".")[0]

            # increment the slice number
            if slice_number == "":
                slice_number = "1"
            else:
                slice_number = str(int(slice_number) + 1)

            slice = slice_number + ".png"

            new_file = os.path.join(case_dir, slice)
            # print(new_file)
            os.rename(old_file, new_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NIfTI files to PNG format")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./model_out/predictions",
        help="Directory containing NIfTI files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./model_out/predictions_png",
        help="Directory to save PNG files",
    )
    parser.add_argument(
        "--width", type=int, default=512, help="Target width for resizing"
    )
    parser.add_argument(
        "--height", type=int, default=512, help="Target height for resizing"
    )

    args = parser.parse_args()

    convert_nifti_to_png(
        args.input_dir, args.output_dir, target_size=(args.width, args.height)
    )

    organise_dir()
