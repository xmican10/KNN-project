import cv2
import os
import argparse
from tqdm import tqdm

## ------ Program parser
parser = argparse.ArgumentParser(description='Convert images in dir to video')
parser.add_argument('--dir', type=str, required=True, 
                    help='Path to dir that contains folders of sequences')
parser.add_argument('--seq', type=str, required=True, 
                    help='Sequence name in <dir>')
parser.add_argument('--fps', type=int, default=3, 
                    help='FPS count (default: 3)')
args = parser.parse_args()

## ------ Set params
image_folder = os.path.join(args.dir, args.seq)
video_name = args.seq + '.mp4'
fps = args.fps

## ------ Create video
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

loop = tqdm(total=len(images), position=0, desc=f"Creating video from {image_folder}")
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))
    loop.update(1)

cv2.destroyAllWindows()
video.release()
