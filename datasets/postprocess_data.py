import numpy as np
import cv2
import os
from PIL import Image
import argparse


def convert_npz_to_png(input_dir, output_dir, target_size=(224, 224)):
    """
    Convert NPZ files to PNG format while preserving the 13 labels.

    Args:
        input_dir: Directory containing NPZ files
        output_dir: Directory to save PNG files
        target_size: Tuple of (width, height) for resizing
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Color mapping for visualization (you can modify these colors)
    # Using a colormap with 13 distinct colors
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

    # Process each NPZ file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".npz"):
            # Load NPZ file
            npz_path = os.path.join(input_dir, filename)
            data = np.load(npz_path)

            # Get the prediction array (assuming it's stored with a specific key)
            # You might need to adjust this depending on your NPZ file structure
            pred = data["pred"] if "pred" in data else data.arr_0

            # Create colored image
            height, width = pred.shape
            colored_image = np.zeros((height, width, 3), dtype=np.uint8)

            # Apply colors based on labels
            for label in range(13):
                mask = pred == label
                colored_image[mask] = colors[label]

            # Resize if necessary
            if (height, width) != target_size:
                colored_image = cv2.resize(
                    colored_image, target_size, interpolation=cv2.INTER_NEAREST
                )

            # Save as PNG
            output_filename = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_dir, output_filename)

            # Save both colored and label versions
            cv2.imwrite(output_path, cv2.cvtColor(colored_image, cv2.COLOR_RGB2BGR))

            # Also save the raw labels as a grayscale PNG
            label_output_path = os.path.join(output_dir, f"labels_{output_filename}")
            resized_pred = cv2.resize(
                pred.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST
            )
            cv2.imwrite(label_output_path, resized_pred)

            print(f"Processed {filename} -> {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NPZ files to PNG format")
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing NPZ files"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save PNG files"
    )
    parser.add_argument(
        "--width", type=int, default=224, help="Target width for resizing"
    )
    parser.add_argument(
        "--height", type=int, default=224, help="Target height for resizing"
    )

    args = parser.parse_args()

    convert_npz_to_png(
        args.input_dir, args.output_dir, target_size=(args.width, args.height)
    )
