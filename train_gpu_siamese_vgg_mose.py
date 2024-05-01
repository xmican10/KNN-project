import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For WSL2: Set Agg backend for generating graphs without GUI
import matplotlib.pyplot as plt
from matplotlib import colors
import argparse
from tqdm import tqdm

import model.unet_siamese_vgg as model

from train.dice_loss import DiceLoss
from mose_loader import MOSEDataset, MOSEValidDataset

## =============================================================================
## ------ Init CUDA
if torch.cuda.is_available():
    print("CUDA is available. Training on GPU.")
else:
    print("CUDA is not available. Training on CPU.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## =============================================================================
## ------ Program parser
parser = argparse.ArgumentParser(description='Train model on DAVIS2016 dataset')
parser.add_argument('--epochs', type=int, default=32, 
                    help='Number of training epochs (default: 32)')
parser.add_argument('--output-pth', type=str, default='model.pth', 
                    help='Output model name (default: \'model.pth)\'')
parser.add_argument('--dataset-root', type=str, default='../MOSE', 
                    help='Root directory for MOSE dataset (default: \'../MOSE)\'')
parser.add_argument('--resume', type=str, default=None, 
                    help='Path to saved model for resuming training. '
                         'If not specified, new model will be trained.')
args = parser.parse_args()

## =============================================================================
## ------ Plot statistics
# Convert torch tensor to numpy array
def tensor_to_numpy(tensor):
    return tensor.squeeze().cpu().detach().numpy()

# Plot images in matrix
def plot_images(frame1, frame2, mask1, mask2, pred_mask1):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    cmap = colors.ListedColormap(['black', 'white'])

    axes[0, 0].set_title('Frames')
    axes[0, 0].imshow((frame1[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)) # Denormalization
    axes[1, 0].imshow((frame2[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    
    axes[0, 1].set_title('GT masks')
    axes[0, 1].imshow(tensor_to_numpy(mask1), cmap=cmap)
    axes[1, 1].imshow(tensor_to_numpy(mask2), cmap=cmap)
    
    axes[0, 2].set_title('Predicted masks')
    axes[0, 2].axis('off')
    axes[1, 2].imshow(tensor_to_numpy(pred_mask1), cmap=cmap)

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
    plt.savefig(args.output_pth+'loss.png')
    plt.close()
        
## =============================================================================
def save_checkpoint(model, filename="model.pth"):
    torch.save(model.state_dict(), filename)
    
## =============================================================================
# --- Set training dataset params
root_dir = args.dataset_root
action = 'train'
batch_size = 16
train_dataset = MOSEDataset(root_dir=root_dir, action=action)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# --- Set validation dataset params
root_dir = args.dataset_root
action = 'valid'
batch_size2 = 16
val_dataset = MOSEDataset(root_dir=root_dir, action=action)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size2, shuffle=True)
## =============================================================================
# --- Set training model params
model = model.VGG11SiameseUnet(pretrained=False).to(device)
if args.resume is not None:
    print("Resume training")
    model.load_state_dict(torch.load(args.resume))
    
# Define loss function and optimizer
loss_f = DiceLoss()
#optimizer = optim.AdamW(model.parameters(), lr=1.3798384386932625e-05, weight_decay=0.0002869552100379619)
optimizer = optim.AdamW(model.parameters(), lr=1e-04, weight_decay=1e-3)

train_loss = []
val_loss = []

## =============================================================================
# --- Training loop
torch.backends.cudnn.benchmark = True
for epoch in range(1, args.epochs+1):    
    ## Run epoch training and validation
    train_running_loss = 0.0
    itrT = 0
    
    model.train()
    loopT = tqdm(total=len(train_dataset)/batch_size, position=0, leave=True, desc=f"Epoch {epoch}/{args.epochs} [Train]")
    for batch_idx, (prev_frame1, curr_frame1, prev_annotation1, curr_annotation1) in enumerate(train_dataloader):
        itrT+=1
        
        prev_frame1 = prev_frame1#.to(device)
        curr_frame1 = curr_frame1#.to(device)
        prev_annotation1 = prev_annotation1#.to(device)
        curr_annotation1 = curr_annotation1#.to(device)

        # Zero the parameter gradients [GPU]
        optimizer.zero_grad(set_to_none=True)

        # Create input tensor
        #input_tensor1 = torch.cat((prev_frame1, curr_frame1, prev_annotation1), dim=1)
        
        # Forward pass 1
        outputs1 = model(prev_frame1, curr_frame1, prev_annotation1)
        
        # Compute loss
        loss = loss_f(outputs1, curr_annotation1).mean()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Log statistics
        train_running_loss += loss.item()
        
        loopT.update(1)
        
    loopT.close()
    
    ## Evaluate model
    if epoch % 4 == 0:
        model.eval()
        itrV = 0
        val_running_loss = 0.0
        loopV = tqdm(total=len(val_dataset)/batch_size2, position=0, leave=True, desc=f"Epoch {epoch}/{args.epochs} [Val]")
        with torch.no_grad():
            for batch_idx, (prev_frame, curr_frame, prev_annotation, curr_annotation) in enumerate(val_dataloader):
                itrV += 1
                
                prev_frame = prev_frame#.to(device)
                curr_frame = curr_frame#.to(device)
                prev_annotation = prev_annotation#.to(device)
                curr_annotation = curr_annotation#.to(device)
                
                #input_tensor = torch.cat((prev_frame, curr_frame, prev_annotation), dim=1)
                outputs = model(prev_frame, curr_frame, prev_annotation)

                loss = loss_f(outputs, curr_annotation).mean()
                val_running_loss += loss.item()
                loopV.update(1)
                
        loopV.close()
    train_loss.append(train_running_loss/itrT)
    val_loss.append(val_running_loss/itrV)
    
    ## Finalize epoch
    # Save checkpoint
    if (epoch % 4 == 0) and (epoch != 0):
        filename = str(epoch)+"_"+args.output_pth
        save_checkpoint(model, filename)
        print(">> Checkpoint of epoch ", str(epoch), " saved.")
    
    # Print and plot statistics
    print('====')
    print('[epoch %5d] train_loss: %.3f, val_loss: %.3f' % (epoch, train_running_loss/itrT, val_running_loss/itrV))
    print('====')
    plot_loss(train_loss, val_loss)

print('Finished Training')
save_checkpoint(model, args.output_pth)
