#%%
# import os
# import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from my_NN_functions import load_data, FeedForwardNN_HP
# from sklearn.metrics import mean_squared_error
# from datetime import date
# date_now = str(date.today())

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% Load data
DOFtype = "Hip3DOF"   # Can be HipKnee4DOF, Hip3DOF, Knee1DOF, KneeAnkle3DOF, Ankle2DOF
data_dir = "..\Data"
Opt_dir = "..\HP_Opt"
fp="/"

features,labels,feature_columns,labels_columns = load_data(data_dir,DOFtype)


# %% Create model
use_tuned_model = True

if use_tuned_model == True:
    study_filename = Opt_dir + fp + DOFtype + '.pkl'
    import pickle
    with open(study_filename, 'rb') as f:
        study = pickle.load(f)

    # Get the best trial
    best_trial = study.best_trial

    # Retrieve Hyperparamters
    num_layers = best_trial.params['num_layers']
    layer_dims = [best_trial.params[f'layer_{i}_dim'] for i in range(num_layers)]
    activation_funcs = [best_trial.params[f'layer_{i}_activation'] for i in range(num_layers)]
    learning_rate = 10**best_trial.params['learning_rate']
    l1_weight = [10**best_trial.params[f'layer_{i}_l1_reg'] for i in range(num_layers)]
    l2_weight = [10**best_trial.params[f'layer_{i}_l2_reg'] for i in range(num_layers)]
    batch_size = best_trial.params['batch_size']

    # l1_weight = 10**best_trial.params["l1_reg"]
    # l2_weight = 10**best_trial.params["l2_reg"]

else:
    num_layers = 2
    layer_dims = [512,256]
    activation_funcs = ['relu','relu']
    learning_rate = 1e-3
    l1_weight = 0
    l2_weight = 1e-4
    batch_size = 512
# model = FeedForwardNN_withReg(features["Train"].shape[-1], n_neurons, labels["Train"].shape[-1], L1, L2)
model = FeedForwardNN_HP(features["Train"].shape[-1], layer_dims, labels["Train"].shape[-1],activation_funcs).to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total trainable parameters:',pytorch_total_params)
# %% Set training parameters and criteria

# Calculate loss weights for outputs
output_ranges = np.ptp(labels["Train"],axis=0)
loss_weights = torch.tensor(np.max(output_ranges)/output_ranges, dtype=torch.float32).to(device)

# Set main training parameters
criterion = nn.L1Loss(reduction='none').to(device)  # L1Loss = MAE, reduction = none allows loss weights and then averaging in training loop
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Set learning rate schedule
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# Early stopping parameters
early_stopping_patience = 20
early_stopping_threshold = 1e-5
best_val_loss = float('inf')
epochs_without_improvement = 0

# Convert data to PyTorch tensors
x_train_tensor = torch.tensor(features["Train"], dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(labels["Train"], dtype=torch.float32).to(device)
x_val_tensor = torch.tensor(features["Val"], dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(labels["Val"], dtype=torch.float32).to(device)

# Create mini-batches
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create DataLoader for validation data
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# %%  Execute training with validation
# Training the model
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()

    train_loss = 0.0
    train_reg_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        # Move batch tensors to GPU
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss = (loss * loss_weights).mean()

        # Add reg loss
        l1_penalty = l1_weight * sum([p.abs().sum() for p in model.parameters()])
        l2_penalty = l2_weight * sum([(p**2).sum() for p in model.parameters()])
        loss_with_penalty = loss + l1_penalty + l2_penalty

        # Backward pass and optimize
        loss_with_penalty.backward()
        # loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_reg_loss += loss_with_penalty.item()

    train_loss /= len(train_dataloader)
    train_reg_loss /= len(train_dataloader)
    total_train_loss = train_loss + train_reg_loss
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4e}, Training Reg Loss: {train_reg_loss:.4e}, Total Training Loss: {total_train_loss:.4e}')
    
        
        # if (epoch+1) % 10 == 0:
        #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluating the model
    model.eval()
    val_loss = 0.0
    val_reg_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation
        for inputs, targets in val_dataloader:
            # Move batch tensors to GPU
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss = (loss * loss_weights).mean()
            # Add reg loss
            l1_penalty = l1_weight * sum([p.abs().sum() for p in model.parameters()])
            l2_penalty = l2_weight * sum([(p**2).sum() for p in model.parameters()])
            loss_with_penalty = loss + l1_penalty + l2_penalty
            
            val_loss += loss.item()
            val_reg_loss += loss_with_penalty.item()
    
    val_loss /= len(val_dataloader)
    val_reg_loss /= len(val_dataloader)
    total_val_loss = val_loss + val_reg_loss
    
    print(f'Validation Loss: {val_loss:.4e}, Validation Reg Loss: {val_reg_loss:.4e}, Total Validation Loss: {total_val_loss:.4e}')

    # Step the scheduler
    scheduler.step(val_loss)

    # Early stopping check
    if  best_val_loss - val_loss > early_stopping_threshold:
        best_val_loss = val_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= early_stopping_patience:
        print(f'Early stopping triggered. No improvement in validation loss for {early_stopping_patience} consecutive epochs.')
        break

print("Training completed.")

# # Validate on test set
# x_test_tensor = torch.tensor(features["Test"], dtype=torch.float32).to(device)
# y_test_tensor = torch.tensor(labels["Test"], dtype=torch.float32).to(device)
# test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# with torch.no_grad():
#     for inputs, targets in test_dataloader:
#         outputs = model(x_test_tensor)
#         mse = mean_squared_error(y_test_tensor.numpy(), outputs.numpy())
#         print(f'Mean Squared Error on test set: {mse:.4f}')

# %% Convert model
libTorch_save_dir = '..\LibTorch_models'
convert_to_script = True
serialise = True
if convert_to_script:
    example = torch.rand([1,features["Train"].shape[-1]]).to(device)
    script_module = torch.jit.trace(model.to(device),example)
    example = torch.rand([1,features["Train"].shape[-1]]).to('cpu')
    script_module_cpu = torch.jit.trace(model.to('cpu'),example)

if serialise:
    script_module.save(libTorch_save_dir+fp+DOFtype+'.pt')
    script_module_cpu.save(libTorch_save_dir+fp+DOFtype+'_cpu.pt')
