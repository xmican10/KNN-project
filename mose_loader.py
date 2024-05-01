import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np

class MOSEDataset(Dataset):
    def __init__(self, root_dir, action, transform=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        action_file = action+'.txt'
        list_file = 'ImageSets/'+action_file
        file = os.path.join(root_dir, list_file)
        
        # Load (image, mask) couples from file in './<root_dir>/ImageSets/'
        with open(file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                frame_path, annotation_path = line.strip().split(' ')
                frame_path = frame_path.lstrip('/')
                annotation_path = annotation_path.lstrip('/')
                
                rel_frame_path = os.path.join(root_dir, frame_path)
                rel_annotation_path = os.path.join(root_dir, annotation_path)
                self.samples.append((rel_frame_path, rel_annotation_path))
        self.n = len(self.samples)

    def __len__(self):
        return len(self.samples) - 1

    def img_to_tensor(self, path, image_size, grayscale=False):
        # Load image in desired color mode
        if grayscale:
            image = Image.open(path).convert('L')
        else:
            image = Image.open(path).convert('RGB')
        
        # Transform PIL Image to tensor
        if grayscale:
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ])
        
        image_tensor = transform(image)
        
        return image_tensor
        
    def get_set(self,idx):
        #print(idx)
        # Get current frame and annotation
        #try:
            curr_frame_path, curr_annotation_path = self.samples[(idx + 1)%self.n] # wraparound indexing
            # Get previous frame and annotation
            prev_frame_path, prev_annotation_path = self.samples[idx]
            return prev_frame_path, curr_frame_path, prev_annotation_path, curr_annotation_path
        #except IndexError:
            #print("Index overflow")
            #self.__getitem__(idx-1)
            #return None, None, None, None
        
    def __getitem__(self, idx, image_size=(128, 128)):
        prev_frame_path, curr_frame_path, prev_annotation_path, curr_annotation_path = self.get_set(idx)
        
        #empty_tensor = torch.empty(128,128).unsqueeze(0)
        #if prev_frame_path is None:
        #    return empty_tensor, empty_tensor, empty_tensor, empty_tensor

        # Check if both frames are from the same sequence
        prev_sequence_name = prev_frame_path.split("/")[-2]
        curr_sequence_name = curr_frame_path.split("/")[-2]
        prev_image_name = (prev_annotation_path.split("/")[-1]).split("_")[0]
        curr_image_name = (curr_annotation_path.split("/")[-1]).split("_")[0]

        if prev_sequence_name != curr_sequence_name or prev_image_name != curr_image_name:
            prev_frame_path, curr_frame_path, prev_annotation_path, curr_annotation_path = self.get_set(idx+1)
            #if prev_frame_path is None:
            #    return empty_tensor, empty_tensor, empty_tensor, empty_tensor

        # Load frames and masks
        curr_frame      = self.img_to_tensor(curr_frame_path, image_size)
        curr_annotation = self.img_to_tensor(curr_annotation_path, image_size, grayscale=True)
        prev_frame      = self.img_to_tensor(prev_frame_path, image_size)
        prev_annotation = self.img_to_tensor(prev_annotation_path, image_size, grayscale=True)

        return prev_frame, curr_frame, prev_annotation, curr_annotation

class MOSEValidDataset(MOSEDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequence = []
        
    def init_seq(self,idx):
        # New sequence
        seq_path, init_mask = self.samples[idx]
        seq = [os.path.join(seq_path, file) for file in os.listdir(seq_path)]
        return seq, init_mask

    def __getitem__(self, idx, image_size=(128, 128)):
        init_mask = torch.empty(128,128).unsqueeze(0)
        if len(self.sequence) < 2:
            self.sequence, init_mask = self.init_seq(idx)
            init_mask = self.img_to_tensor(init_mask, image_size, grayscale=True)
            
        frame1 = self.img_to_tensor(self.sequence.pop(0), image_size)
        frame2 = self.img_to_tensor(self.sequence.pop(0), image_size)

        return init_mask, frame1, frame2
