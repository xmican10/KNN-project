import numpy as np
from PIL import Image
import metrics
import argparse
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from davis_loader import DAVIS2016Dataset
## =============================================================================
# DAVIS2016 Leaderborard: https://paperswithcode.com/sota/visual-object-tracking-on-davis-2016
## =============================================================================
## ------ Program parser
parser = argparse.ArgumentParser(description='Evaluation script for DAVIS2016 dataset')
parser.add_argument('--dataset-root', type=str, default='./DAVIS', 
                    help='Root directory for DAVIS2016 dataset (default: \'./DAVIS)\'')
parser.add_argument('--res-dir', type=str, default=None, 
                    help='Path to eval_res output dir, if not specified, eval_res.py will be runed.')
parser.add_argument('--model', type=str, required=False, default=None, 
                    help='Input .pth model, specify if res-dir is not specified.')
parser.add_argument('--mode', type=str, choices=['train', 'val', 'trainval'], default='val',
                    help='Specifies the DAVIS2016 dataset split: test, val (default), or train, specify if res-dir is not specified.')
args = parser.parse_args()

if args.res_dir is None:
    if args.model is None:
        print(f"ERROR: Input model .pth was not specified, cannot run the evaluation. (Specify [--res-dir] or add [--model])!")
        sys.exit(1)

img_size = (128,128)

# --- Set validation dataset params
root_dir = args.dataset_root
action = args.mode
dataset = DAVIS2016Dataset(root_dir=root_dir, action=action)
num_samples = len(dataset.samples)

# --- Load dir with moodel output images
outputs = []
for root, dirs, files in os.walk(args.res_dir):
    for file in files:
        output = os.path.relpath(os.path.join(root, file), args.res_dir)
        outputs.append(os.path.join(args.res_dir, output))
num_outputs = len(outputs)

if num_samples != num_outputs:
    print(f"ERROR: {args.res_dir} contains unexpected number of samples {num_outputs}")
    sys.exit(1)

# --- Run evaluation
iou = f = 0
loop = tqdm(total=num_samples, position=0, desc=f"Evaluation")
for i in range(num_samples):
    # Groundtruth mask
    g = np.array(Image.open(dataset.samples[i][1]).convert('L').resize(img_size, Image.LANCZOS))
    # Predicted mask
    m = np.array(Image.open(outputs[i]).convert('L'))
    
    iou += metrics.jaccard(g, m)
    f += metrics.db_eval_boundary(m, g)
    
    loop.update(1)

iou_avg = iou/num_samples
f_avg = f/num_samples
print("Statistics:")
print(">> J mean: ", iou_avg.item() * 100, "%")
print(">> F mean: ", f_avg.item() * 100, "%")
