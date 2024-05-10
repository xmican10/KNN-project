import os
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

import model.unet_siamese_vgg11 as model
from train.dice_loss import DiceLoss
from mose_loader import MOSEDataset

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
parser.add_argument('--batch', type=int, default=256, 
                    help='Batch size')
parser.add_argument('--num_workers', type=int, default=0, 
                    help='Number of workers')
parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate")    
parser.add_argument("--w_decay", type=float, default=0.0001, help="Weight decay")

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
batch_size = args.batch
train_dataset = MOSEDataset(root_dir=root_dir, action=action)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

# --- Set validation dataset params
root_dir = args.dataset_root
action = 'valid'
batch_size2 = args.batch
val_dataset = MOSEDataset(root_dir=root_dir, action=action)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size2, shuffle=True, pin_memory=True, num_workers=args.num_workers)
## =============================================================================
# --- Set training model params
model = model.VGG11SiameseUnet().to(device)
if args.resume is not None:
    print("Resume training")
    model.load_state_dict(torch.load(args.resume))
    
# Define loss function and optimizer
loss_f = DiceLoss()
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.w_decay)

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

        prev_frame1 = prev_frame1.to(device, non_blocking=True)
        curr_frame1 = curr_frame1.to(device, non_blocking=True)
        prev_annotation1 = prev_annotation1.to(device, non_blocking=True)
        curr_annotation1 = curr_annotation1.to(device, non_blocking=True)

        # Zero the parameter gradients [GPU]
        optimizer.zero_grad(set_to_none=True)
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

                prev_frame = prev_frame.to(device, non_blocking=True)
                curr_frame = curr_frame.to(device, non_blocking=True)
                prev_annotation = prev_annotation.to(device, non_blocking=True)
                curr_annotation = curr_annotation.to(device, non_blocking=True)
                outputs = model(prev_frame, curr_frame, prev_annotation)

                loss = loss_f(outputs, curr_annotation).mean()
                val_running_loss += loss.item()
                loopV.update(1)
        val_loss.append(val_running_loss/itrV)
        train_loss.append(train_running_loss/itrT)
        print('[epoch %5d] val_loss: %.3f' % (epoch, val_running_loss/itrV))
        plot_loss(train_loss, val_loss)
        loopV.close()


    ## Finalize epoch
    # Save checkpoint
    if (epoch % 4 == 0) and (epoch != 0):
        directory, filename_with_extension = os.path.split(args.output_pth)
        filename, extension = os.path.splitext(filename_with_extension)
        modified_filename = f"{filename}_epoch-{epoch}{extension}"
        file_path = os.path.join(directory, modified_filename)

        #filename = str(epoch)+"_"+args.output_pth
        save_checkpoint(model, file_path)
        print(">> Checkpoint of epoch ", str(epoch), " saved.")
