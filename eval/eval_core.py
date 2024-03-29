import os
import sys
import torch
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from davis_loader import DAVIS2016Dataset
import model.unet as unet

class Eval():
    def __init__(self, args):
        self.output_dir = args.output_dir
        # --- Set validation dataset params
        root_dir = args.dataset_root
        action = args.mode
        self.dataset = DAVIS2016Dataset(root_dir=root_dir, action=action)
        
        # --- Load model
        self.model = unet.UNet(n_channels=7, n_classes=1)
        self.model.load_state_dict(torch.load(args.model))
        self.model.eval()

    def tensor_to_numpy(self, tensor):
        # --- Convert torch tensor to numpy array
        return tensor.squeeze().cpu().detach().numpy()

    def get_sample_dir(self, i):
        # Get path and split it to parts
        path = self.dataset.samples[i][0]
        parts = os.path.split(os.path.dirname(path))
        # Get directory name
        last_folder_name = parts[-1]

        return last_folder_name
    
    def plot_images(self, ref, out, i, sample_dir):
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        cmap = colors.ListedColormap(['black', 'white'])
        #cmap='binary_r'
        axes[0].imshow(self.tensor_to_numpy(ref), cmap=cmap)
        axes[0].set_title('Expected mask')
        axes[1].imshow(self.tensor_to_numpy(out), cmap=cmap)
        axes[1].set_title('Predicted mask')

        # Prepare output dir
        output_dir = os.path.join(self.output_dir, sample_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.join(output_dir, f"{i}.png")

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
    def save_res(self, res, i, sample_dir):
        img_array = self.tensor_to_numpy(res)
        # Prepare output dir
        output_dir = os.path.join(self.output_dir, sample_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.join(output_dir, f"{i}.png")
    
        threshold = 0.5
        data_binary = (img_array > threshold).astype(np.uint8)
        data_normalized = data_binary * 255

        image = Image.fromarray(data_normalized, 'L')
        image.save(filename)
        
    def run(self, compare = True):
        # --- Evaluation loop
        last_dir = new_dir = self.get_sample_dir(0)
        _, prev_frame, curr_frame, prev_annotation, curr_annotation = self.dataset[0]
        input_tensor = torch.cat((prev_frame, curr_frame, prev_annotation), dim=1)
        outputs = self.model(input_tensor)
        if compare:
            self.plot_images(curr_annotation, outputs, 1, last_dir)
        else:
                self.save_res(outputs, 1, last_dir)
        frame_cnt = 2
        for i in range(len(self.dataset)):
            new_dir = self.get_sample_dir(i)
            next_frame, prev_frame, curr_frame, prev_annotation, curr_annotation = self.dataset[i]
            if next_frame:
                # Get next couple of frames, because currently prev_frame is from different sequence than curr_frame
                frame_cnt = 0
                continue
            if frame_cnt == 0:
                # New input sequence
                input_tensor = torch.cat((prev_frame, curr_frame, prev_annotation), dim=1)
                frame_cnt += 1
            else:
                input_tensor = torch.cat((prev_frame, curr_frame, outputs), dim=1)
            outputs = self.model(input_tensor)
            last_dir = new_dir
            if compare:
                self.plot_images(curr_annotation, outputs, frame_cnt, last_dir)
            else:
                self.save_res(outputs, frame_cnt, last_dir)
            frame_cnt += 1
            
        print('Finished Evaluation')
