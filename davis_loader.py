import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class DAVIS2016Dataset(Dataset):
    def __init__(self, root_dir, action, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        action_file = action+'.txt'
        list_file = 'ImageSets/480p/'+action_file
        file = os.path.join(root_dir, list_file)
        
        # Load (image, mask) couples from file in './<root_dir>/ImageSets/480p/'
        with open(file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                frame_path, annotation_path = line.strip().split(' ')
                frame_path = frame_path.lstrip('/')
                annotation_path = annotation_path.lstrip('/')
                
                rel_frame_path = os.path.join(root_dir, frame_path)
                rel_annotation_path = os.path.join(root_dir, annotation_path)
                self.samples.append((rel_frame_path, rel_annotation_path))

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
        
        # Apply transform to image
        image_tensor = transform(image)
        # Add batch dimension = 1
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
        
    def get_set(self,idx):
        # Get current frame and annotation
        curr_frame_path, curr_annotation_path = self.samples[idx + 1]
        # Get previous frame and annotation
        prev_frame_path, prev_annotation_path = self.samples[idx]
        return prev_frame_path, curr_frame_path, prev_annotation_path, curr_annotation_path
        
    def __getitem__(self, idx, image_size=(128, 128)):
        next_frame = False
        prev_frame_path, curr_frame_path, prev_annotation_path, curr_annotation_path = self.get_set(idx)

        # Check if both frames are from the same sequence
        prev_sequence_name = prev_frame_path.split("/")[4]
        curr_sequence_name = curr_frame_path.split("/")[4]
        if prev_sequence_name != curr_sequence_name:
            next_frame = True
            return next_frame, torch.zeros(1,1), torch.zeros(1,1), torch.zeros(1,1), torch.zeros(1,1)
            #prev_frame_path, curr_frame_path, prev_annotation_path, curr_annotation_path = self.get_set(idx+1)

        # Load frames and masks
        curr_frame      = self.img_to_tensor(curr_frame_path, image_size)
        curr_annotation = self.img_to_tensor(curr_annotation_path, image_size, grayscale=True)
        prev_frame      = self.img_to_tensor(prev_frame_path, image_size)
        prev_annotation = self.img_to_tensor(prev_annotation_path, image_size, grayscale=True)

        if self.transform:
            prev_frame, prev_annotation, curr_frame, curr_annotation = self.transform(prev_frame, prev_annotation, curr_frame, curr_annotation)

        return next_frame, prev_frame, curr_frame, prev_annotation, curr_annotation
