import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Pro WSL2: Nastavení Agg backendu pro generování grafů bez GUI
import matplotlib.pyplot as plt
from matplotlib import colors
import argparse

import unet
from davisLoader import DAVIS2016Dataset

## =============================================================================
## ------ Program parser
parser = argparse.ArgumentParser(description='Train model on DAVIS2016 dataset')
parser.add_argument('--epochs', type=int, default=8, 
                    help='Number of training epochs (default: 8)')
parser.add_argument('--dataset-root', type=str, default='./davis', 
                    help='Root directory for DAVIS2016 dataset (default: \'./davis)\'')
args = parser.parse_args()

## =============================================================================
## ------ Plot statistics
# Convert torch tensor to numpy array
def tensor_to_numpy(tensor):
    return tensor.squeeze().cpu().detach().numpy()

# Plot images in matrix
def plot_images(img1, img2, img3, mask_pred, mask_true):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow((img1[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)) # Denormalization
    axes[0, 0].set_title('Previous frame')
    axes[1, 0].imshow((img2[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    axes[1, 0].set_title('Current frame')
    
    axes[0, 1].imshow(tensor_to_numpy(img3), cmap=colors.ListedColormap(['black', 'white']))
    axes[0, 1].set_title('Previous mask')
    axes[1, 1].imshow(tensor_to_numpy(mask_true), cmap=colors.ListedColormap(['black', 'white']))
    axes[1, 1].set_title('Expected mask (target)')
    axes[1, 2].imshow(tensor_to_numpy(mask_pred), cmap=colors.ListedColormap(['black', 'white']))
    axes[1, 2].set_title('Predicted Mask')
    # alternative: cmap='binary_r'

    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('output.png')
    plt.close()

# Plot losses
def plot_loss(train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_loss)+1), train_loss, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.plot(range(1, len(val_loss)+1), val_loss, marker='o', linestyle='-', color='orange', label='Validation Loss')
    plt.title("Training vs. Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('loss.png')
    plt.close()
    
## =============================================================================
# --- Set training dataset params
root_dir = './davis'
action = 'train'
train_dataset = DAVIS2016Dataset(root_dir=root_dir, action=action)

# --- Set validation dataset params
root_dir = './davis'
action = 'val'
val_dataset = DAVIS2016Dataset(root_dir=root_dir, action=action)

## =============================================================================
# --- Set training model params
n_channels = 7  # 3 (RGB) + 3 (RGB) + 1 (Grayscale)
n_classes = 1  # Binary (background/foreground) segmentation
model = unet.UNet(n_channels=n_channels, n_classes=n_classes)

# Define loss function and optimizer
loss_f = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-5)

# criterion = nn.BCEWithLogitsLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)

train_loss = []
val_loss = []

# --- Training loop
for epoch in range(args.epochs):
    train_running_loss = 0.0
    val_running_loss = 0.0
    model.train()
    for i in range(len(train_dataset)):
        prev_frame, curr_frame, prev_annotation, curr_annotation = train_dataset[i]
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Create input tensor
        input_tensor = torch.cat((prev_frame, curr_frame, prev_annotation), dim=1)

        # Forward pass
        outputs = model(input_tensor)
        
        loss_f = torch.nn.BCELoss()
        loss = loss_f(outputs, curr_annotation)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print statistics
        train_running_loss += loss.item()
        
        if i % 10 == 0:  # Print every Nth batch
            print('[%5d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, loss.item()))

            # Plot for monitoring
            plot_images(prev_frame, curr_frame, prev_annotation, outputs, curr_annotation)
            
    # Evaluate model
    model.eval()
    with torch.no_grad():
        for i in range(len(val_dataset)):
            prev_frame, curr_frame, prev_annotation, curr_annotation = val_dataset[i]
            
            input_tensor = torch.cat((prev_frame, curr_frame, prev_annotation), dim=1)
            outputs = model(input_tensor)

            loss = torch.nn.BCELoss()(outputs, curr_annotation)
            val_running_loss += loss.item()
            
    train_loss.append(train_running_loss/len(train_dataset))
    val_loss.append(val_running_loss/len(val_dataset))
    
    print('====')
    print('[epoch %5d] train_loss: %.3f, val_loss: %.3f' %
                  (epoch + 1, train_running_loss/len(train_dataset), val_running_loss/len(val_dataset)))
    print('====')
    plot_loss(train_loss, val_loss)

            
print('Finished Training')
torch.save(model.state_dict(), 'unet_model.pth')
