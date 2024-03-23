import os
from PIL import Image
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
                # Predpokladá sa, že cesty v train.txt už sú relatívne k root_dir
                frame_path, annotation_path = line.strip().split(' ')
                frame_path = frame_path.lstrip('/')
                annotation_path = annotation_path.lstrip('/')
                # Prídavok root_dir k získaniu absolútnej cesty
                rel_frame_path = os.path.join(root_dir, frame_path)
                rel_annotation_path = os.path.join(root_dir, annotation_path)
                self.samples.append((rel_frame_path, rel_annotation_path))

    def __len__(self):
        return len(self.samples) - 1

    def img_to_tensor(self, path, grayscale=False, image_size=(128, 128),):
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
        
    def __getitem__(self, idx):
        # Get current frame and annotation
        curr_frame_path, curr_annotation_path = self.samples[idx + 1]
        # Get previous frame and annotation
        prev_frame_path, prev_annotation_path = self.samples[idx]

        # Načtení snímků a masek
        curr_frame      = self.img_to_tensor(curr_frame_path)
        curr_annotation = self.img_to_tensor(curr_annotation_path, grayscale=True)
        prev_frame      = self.img_to_tensor(prev_frame_path)
        prev_annotation = self.img_to_tensor(prev_annotation_path, grayscale=True)

        if self.transform:
            # Předpokládejme, že transformace dokáže zpracovat obě dvojice současně
            prev_frame, prev_annotation, curr_frame, curr_annotation = self.transform(prev_frame, prev_annotation, curr_frame, curr_annotation)

        return prev_frame, curr_frame, prev_annotation, curr_annotation
