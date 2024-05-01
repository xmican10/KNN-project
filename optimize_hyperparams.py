import optuna
import torch
import torch.optim as optim
from torch import nn
from torchvision import datasets, transforms
import model.unet_siamese_vgg as model
from train.dice_loss import DiceLoss
from mose_loader import MOSEDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
## =============================================================================
## ------ Init CUDA
if torch.cuda.is_available():
    print("CUDA is available. Training on GPU.")
else:
    print("CUDA is not available. Training on CPU.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## =============================================================================
# --- Set training dataset params
root_dir = "../MOSE"
action = 'train_mini_obj2'
batch_size = 48
train_dataset = MOSEDataset(root_dir=root_dir, action=action)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# --- Set validation dataset params
root_dir = "../MOSE"
action = 'valid_mini_obj4'
batch_size2 = 48
val_dataset = MOSEDataset(root_dir=root_dir, action=action)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size2, shuffle=True, pin_memory=True)
## =============================================================================
# --- Set training model params
model = model.VGG11SiameseUnet(pretrained=False).to(device)
    
# Define loss function and optimizer
loss_f = DiceLoss()
#optimizer = optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-3)

train_loss = []
val_loss = []
## =============================================================================

def train(model, optimizer, epoch):
    train_running_loss = 0.0
    itrT = 0
    model.train()
    loopT = tqdm(total=len(train_dataset)/batch_size, position=0, leave=True, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (prev_frame1, curr_frame1, prev_annotation1, curr_annotation1) in enumerate(train_dataloader):
        #if prev_frame1 is None:
        #    continue
        
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
    return train_running_loss/itrT

def validate(model):
    val_running_loss = 0.0
    itrV = 0
    model.eval()
    loopV = tqdm(total=len(val_dataset)/batch_size2, position=0, leave=True, desc=f"[Val]")
    with torch.no_grad():
        for batch_idx, (prev_frame, curr_frame, prev_annotation, curr_annotation) in enumerate(val_dataloader):
            #if prev_frame.nelement() == 0:
            #    continue
            
            itrV += 1
            
            prev_frame = prev_frame.to(device, non_blocking=True)
            curr_frame = curr_frame.to(device, non_blocking=True)
            prev_annotation = prev_annotation.to(device, non_blocking=True)
            curr_annotation = curr_annotation.to(device, non_blocking=True)
            
            #input_tensor = torch.cat((prev_frame, curr_frame, prev_annotation), dim=1)
            outputs = model(prev_frame, curr_frame, prev_annotation)

            loss = loss_f(outputs, curr_annotation).mean()
            val_running_loss += loss.item()
            loopV.update(1)
            
    loopV.close()
    return val_running_loss/itrV

## =============================================================================
def save_checkpoint(model, filename="model.pth"):
    torch.save(model.state_dict(), filename)
    
## =============================================================================

def objective(trial):
    # Definice hyperparametrů
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    #optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'RMSprop'])

    #model = MyModel()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)#getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    
    for epoch in range(8):
        loss = train(model, optimizer, epoch)  # Funkce pro trénování modelu
        print(f"[Epoch {epoch}] Train loss: {loss}")
    val_loss = validate(model)  # Funkce pro validaci modelu
    print(f"Val loss: {val_loss}")
    save_checkpoint(model, filename=f"model_{lr}_{weight_decay}.pth")
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

print("Best hyperparameters: {}".format(study.best_trial.params))

## Plot visualisations
fig = optuna.visualization.plot_param_importances(study)
fig.write_image("hyperparam_importance.png")
#plt.close(fig)

fig = optuna.visualization.plot_optimization_history(study)
fig.write_image("optimization_history.png")
#plt.close(fig)

fig = optuna.visualization.plot_parallel_coordinate(study)
fig.write_image("optimization_history_pc.png")
#plt.close(fig)

fig = optuna.visualization.plot_slice(study, params=['lr', 'weight_decay'])
fig.write_image("optimization_history_pc_sliced.png")
#plt.close(fig)

# Trial 19 finished with value: 0.05664771461729393 and parameters: {'lr': 0.000879626769220│                                                         │  19618 eva        20   0  1.1T  282M  121M S  0.0  1.2  0:0899, 'weight_decay': 0.0009409430628759515}. Best is trial 19 with value: 0.05664771461729393.                       │                                                         │  19673 eva        20   0  1.1T  282M  121M S  0.0  1.2  0:Best hyperparameters: {'lr': 0.0008796267692200899, 'weight_decay': 0.0009409430628759515}
