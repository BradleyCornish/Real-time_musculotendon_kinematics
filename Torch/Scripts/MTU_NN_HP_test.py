#%%
import os
# import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.trial import TrialState
import numpy as np
# import pandas as pd
from datetime import date
from my_NN_functions import load_data, FeedForwardNN_HP, load_study, save_study #FeedForwardNN_withReg
date_now = str(date.today())

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% set directories and load data
DOFtype = "Hip3DOF"   # Can be HipKnee4DOF, Hip3DOF, Knee1DOF, KneeAnkle3DOF, Ankle2DOF
data_dir = "..\Data"
Opt_dir = "..\HP_Opt"
fp="/"

features,labels,feature_columns,labels_columns = load_data(data_dir,DOFtype)

# %% Set training paramters and criteria, and create tensors

# Calculate loss weights for outputs
output_ranges = np.ptp(labels["Train"],axis=0)
loss_weights = torch.tensor(np.max(output_ranges)/output_ranges, dtype=torch.float32).to(device)

# Convert data to PyTorch tensors
x_train_tensor = torch.tensor(features["Train"], dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(labels["Train"], dtype=torch.float32).to(device)
x_val_tensor = torch.tensor(features["Val"], dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(labels["Val"], dtype=torch.float32).to(device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %% 
# Hyperparameter tuning objective function

def objective(trial):
    # Define the hyperparameters to tune
    input_dim = features["Train"].shape[-1]
    output_dim = labels["Train"].shape[-1]

    # Hyperparameters to tune
    num_layers = trial.suggest_int('num_layers', 1, 3)
    layer_dims = [trial.suggest_int(f'layer_{i}_dim', 64, 512) for i in range(num_layers)]
    activation_funcs = [trial.suggest_categorical(f'layer_{i}_activation', ['relu', 'tanh']) for i in range(num_layers)]
    batch_size = trial.suggest_int('batch_size', 64, 1024,log=True)
    learning_rate = 10**trial.suggest_int('learning_rate', -5, -2)
    l1_weight = [10**trial.suggest_int(f'layer_{i}_l1_reg', -6, -3) for i in range(num_layers)]
    l2_weight = [10**trial.suggest_int(f'layer_{i}_l2_reg', -6, -3) for i in range(num_layers)]

    # When using a single weight for all layers:
    # l1_weight = 10**trial.suggest_int('l1_reg', -6, -3)
    # l2_weight = 10**trial.suggest_int('l2_reg', -6, -3)

    model = FeedForwardNN_HP(input_dim, layer_dims, output_dim, activation_funcs, l1_weight, l2_weight).to(device)
    criterion = nn.L1Loss(reduction='none').to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True,threshold=5e-5)

    # Define earley stopping criteria
    early_stopping_patience = 20
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    early_stopping_threshold = 2e-5

    # Add model size penalty to avoid excessivley large models that offer little improvement
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_penalty = 2e-10 * total_trainable_params

    # Create mini-batches
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create DataLoader for validation data
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_epochs = 500
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

            # # Add reg loss old (moved to function in model)
            # l1_penalty = l1_weight * sum([p.abs().sum() for p in model.parameters()])
            # l2_penalty = l2_weight * sum([(p**2).sum() for p in model.parameters()])
            # loss_with_penalty = loss + l1_penalty + l2_penalty

            # Add reg loss new
            regularization_loss = model.compute_regularization_loss()
            loss_with_penalty = loss + regularization_loss

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

        # Evaluate model on val set
        model.eval()
        val_loss = 0.0
        val_reg_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                # Move batch tensors to GPU
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = (loss * loss_weights).mean()

                # Add reg loss 
                regularization_loss = model.compute_regularization_loss()
                loss_with_penalty = loss + regularization_loss
                
                val_loss += loss.item()
                val_reg_loss += loss_with_penalty.item()
        val_loss /= len(val_dataloader)
        val_reg_loss /= len(val_dataloader)
        total_val_loss = val_loss + val_reg_loss

        print(f'Validation Loss: {val_loss:.4e}, Validation Reg Loss: {val_reg_loss:.4e}, Total Validation Loss: {total_val_loss:.4e}')
        # print(f'current LR: {scheduler.get_last_lr(): .3e}')
        # Step the scheduler
        scheduler.step(total_val_loss)
        trial.report(total_val_loss, epoch)

        # prune worst trials
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Early stopping check
        include_reg_loss_in_opt = False
        if include_reg_loss_in_opt:
            opt_loss = total_val_loss
        else: 
            opt_loss = val_loss

        if best_val_loss - opt_loss > early_stopping_threshold:
            best_val_loss = opt_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= early_stopping_patience:
            break
    return best_val_loss + model_size_penalty


def save_study_callback(study, trial):
    save_study(study, study_filename)

# Create a study or load study if previously created
study_filename = Opt_dir + fp + DOFtype + '.pkl'

if os.path.exists(study_filename):
    study = load_study(study_filename)
else:
    study = optuna.create_study(direction='minimize',sampler= optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner(),)
    os.makedirs(Opt_dir,exist_ok=True)
# Perform HP optimisation
study.optimize(objective, n_trials=200,callbacks=[save_study_callback])

# Save the study
save_study(study, study_filename)

# Print the best hyperparameters
print('Best hyperparameters: ', study.best_params)

# Print the study statistics
print('Number of finished trials: ', len(study.trials))
print('Best trial:')
trial = study.best_trial

print(f'  Value: {trial.value}')

print('  Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')