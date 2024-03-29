import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Pro WSL2: Nastavení Agg backendu pro generování grafů bez GUI
import matplotlib.pyplot as plt
from matplotlib import colors
import argparse
from tqdm import tqdm

import model.unet2 as unet
import model.deeplabv3 as deeplabv3

from train.dice_loss import DiceLoss
from davis_loader import DAVIS2016Dataset
import dataset_shuffle

if torch.cuda.is_available():
    print("CUDA is available. Training on GPU.")
else:
    print("CUDA is not available. Training on CPU.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## =============================================================================
## ------ Program parser
parser = argparse.ArgumentParser(description='Train model on DAVIS2016 dataset')
parser.add_argument('--epochs', type=int, default=8, 
                    help='Number of training epochs (default: 8)')
parser.add_argument('--output-pth', type=str, default='model.pth', 
                    help='Output model name (default: \'model.pth)\'')
parser.add_argument('--dataset-root', type=str, default='./DAVIS', 
                    help='Root directory for DAVIS2016 dataset (default: \'./DAVIS)\'')
parser.add_argument('--resume', type=str, default=None, 
                    help='Path to saved model for resuming training. '
                         'If not specified, new model will be trained.')
parser.add_argument('--shuffle', action='store_true',
                    help='Enable shuffling of the DAVIS2016 dataset.')
args = parser.parse_args()

## =============================================================================
## ------ Plot statistics
# Convert torch tensor to numpy array
def tensor_to_numpy(tensor):
    return tensor.squeeze().cpu().detach().numpy()

# Plot images in matrix
def plot_images(frame1, frame2, frame3, mask1, mask2, mask3, pred_mask1, pred_mask2):
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    cmap = colors.ListedColormap(['black', 'white'])

    axes[0, 0].set_title('Frames')
    axes[0, 0].imshow((frame1[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)) # Denormalization
    axes[1, 0].imshow((frame2[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    axes[2, 0].imshow((frame3[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    
    axes[0, 1].set_title('GT masks')
    axes[0, 1].imshow(tensor_to_numpy(mask1), cmap=cmap)
    axes[1, 1].imshow(tensor_to_numpy(mask2), cmap=cmap)
    axes[2, 1].imshow(tensor_to_numpy(mask3), cmap=cmap)
    
    axes[0, 2].set_title('Predicted masks')
    axes[0, 2].axis('off')
    axes[1, 2].imshow(tensor_to_numpy(pred_mask1), cmap=cmap)
    axes[2, 2].imshow(tensor_to_numpy(pred_mask2), cmap=cmap)

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
def save_checkpoint(model, filename="model.pth"):
    torch.save(model.state_dict(), filename)
    
## =============================================================================
# --- Set training dataset params
root_dir = args.dataset_root
#action = 'train_shuffled'
action = 'train'
train_dataset = DAVIS2016Dataset(root_dir=root_dir, action=action)

# --- Set validation dataset params
root_dir = args.dataset_root
action = 'val'
val_dataset = DAVIS2016Dataset(root_dir=root_dir, action=action)

## =============================================================================
# --- Set training model params
model = unet.UNet(n_channels=7, n_classes=1).to(device)
#model = deeplabv3.model(num_classes=1, in_channels=7)
#deeplabv3.replace_bn_with_gn(model)
if args.resume is not None:
    print("Resume training")
    model.load_state_dict(torch.load(args.resume))
# Define loss function and optimizer
# loss_f = nn.BCELoss()
#optimizer = optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-5)
loss_f = DiceLoss()
optimizer = optim.AdamW(model.parameters(), lr=2.5e-5, weight_decay=1e-5)

# criterion = nn.BCEWithLogitsLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)

train_loss = []
val_loss = []

# --- Training loop
for epoch in range(1, args.epochs+1):
    ## Shuffle dataset after each epoch
    if args.shuffle:
        print(">> Shuffling dataset")
        dataset_shuffle.shuffle()
        action = 'train_shuffled'
        train_dataset = DAVIS2016Dataset(root_dir=root_dir, action=action)
    
    ## Run epoch training and validation
    train_running_loss = 0.0
    val_running_loss = 0.0
    itr = 0
    
    model.train()
    loopT = tqdm(total=len(train_dataset), position=0, leave=True, desc=f"Epoch {epoch}/{args.epochs} [Train]")
    for i in range(len(train_dataset)-1):
        next_frame1, prev_frame1, curr_frame1, prev_annotation1, curr_annotation1 = train_dataset[i]
        next_frame2, prev_frame2, curr_frame2, prev_annotation2, curr_annotation2 = train_dataset[i+1]
        if next_frame1 or next_frame2:
            continue
        prev_frame1 = prev_frame1.to(device)
        curr_frame1 = curr_frame1.to(device)
        prev_annotation1 = prev_annotation1.to(device)
        curr_annotation1 = curr_annotation1.to(device)
        prev_frame2 = prev_frame2.to(device)
        curr_frame2 = curr_frame2.to(device)
        prev_annotation2 = prev_annotation2.to(device)
        curr_annotation2 = curr_annotation2.to(device)
        itr+=1
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Create input tensor1
        input_tensor1 = torch.cat((prev_frame1, curr_frame1, prev_annotation1), dim=1)

        # Forward pass 1
        outputs1 = model(input_tensor1)
        
        # Compute loss
        loss1 = loss_f(outputs1, curr_annotation1)
        
        # Create input tensor2
        input_tensor2 = torch.cat((prev_frame2, curr_frame2, outputs1), dim=1)

        # Forward pass 1
        outputs2 = model(input_tensor2)
        
        # Compute loss
        loss = loss_f(outputs2, curr_annotation2)+loss1
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print statistics
        train_running_loss += loss.item()
        
        if i % 10 == 0:  # Print every Nth batch
            loopT.set_postfix(loss=loss.item())
            # Plot for monitoring
            plot_images(prev_frame1.cpu(), curr_frame1.cpu(), curr_frame2.cpu(), prev_annotation1.cpu(), curr_annotation1.cpu(), prev_annotation2.cpu(), outputs1.cpu(), outputs2.cpu())
        loopT.update(1)
        
    loopT.close()
    
    ## Evaluate model
    model.eval()
    loopV = tqdm(total=len(val_dataset), position=0, leave=True, desc=f"Epoch {epoch}/{args.epochs} [Val]")
    with torch.no_grad():
        for i in range(len(val_dataset)):
            next_frame, prev_frame, curr_frame, prev_annotation, curr_annotation = val_dataset[i]
            if next_frame:
                continue
            prev_frame = prev_frame.to(device)
            curr_frame = curr_frame.to(device)
            prev_annotation = prev_annotation.to(device)
            curr_annotation = curr_annotation.to(device)
            
            input_tensor = torch.cat((prev_frame, curr_frame, prev_annotation), dim=1)
            outputs = model(input_tensor)

            loss = loss_f(outputs, curr_annotation)
            val_running_loss += loss.item()
            loopV.update(1)
            
    loopV.close()
    train_loss.append(train_running_loss/itr)
    val_loss.append(val_running_loss/len(val_dataset))
    
    ## Finalize epoch
    # Save checkpoint
    if (epoch % 4 == 0) and (epoch != 0):
        filename = str(epoch)+"_"+args.output_pth
        save_checkpoint(model, filename)
        print(">> Checkpoint of epoch ", str(epoch), " saved.")
    
    # Print and plot stats
    print('====')
    print('[epoch %5d] train_loss: %.3f, val_loss: %.3f' % (epoch, train_running_loss/len(train_dataset), val_running_loss/len(val_dataset)))
    print('====')
    plot_loss(train_loss, val_loss)
            
print('Finished Training')
save_checkpoint(model, args.output_pth)
