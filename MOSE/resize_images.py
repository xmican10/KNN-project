import sys
from PIL import Image
import os

def resize_images(directory, target_size=(256, 256)):
    """
    Resize all jpg and png images in the given directory and its subdirectories to the target size and replace the original images.
    
    Args:
    - directory (str): The directory containing the images.
    - target_size (tuple): The target size for the images. Default is (128, 128).
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    image_path = os.path.join(root, file)
                    with Image.open(image_path) as img:
                        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                        # Overwrite the original image with the resized one
                        img_resized.save(image_path)
                        print(f"Resized {image_path} to {target_size} and replaced the original image.")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    # Check if directory path is provided as command line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]
    if not os.path.isdir(directory_path):
        print("Invalid directory path provided.")
        sys.exit(1)

    resize_images(directory_path)

