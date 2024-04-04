import cv2
import os
import imageio
import argparse
from tqdm import tqdm

# ------ Program parser
parser = argparse.ArgumentParser(description='Convert images in dir to video')
parser.add_argument('--dir', type=str, required=True, help='Path to dir that contains folders of sequences')
parser.add_argument('--seq', type=str, required=True, help='Sequence name in <dir>')
parser.add_argument('--fps', type=int, default=3, help='FPS count (default: 3)')
args = parser.parse_args()

# ------ Set params
image_folder = os.path.join(args.dir, args.seq)
gif_name = args.seq + '.gif'
fps = args.fps

# ------ Create GIF
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()

frames = []
for image in tqdm(images, desc=f"Creating GIF from {image_folder}"):
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)

# Determine GIF duration based on FPS
duration = 1 / fps

# Save the frames as a GIF
imageio.mimsave(gif_name, frames, 'GIF', duration=duration)

print(f"GIF saved as {gif_name}")
