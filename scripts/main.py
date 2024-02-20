#%%
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pingouin as pg
from scipy import stats as st
from functions import editBAAnnotation, get_features_and_labels, get_output_labels, organise_results, get_Muscles_DOFs_Bodies_ScaleFactors

fsep="/"
NN_model_path = "../NN_models"
MTU_data_path = "../data"

# Select whether to load the optimised model architecture or build a new model from scratch 
# If loading optimised model, select whether to retrain model weights from scratch
use_optimised_model = True
retrain_model = False

# Select OpenSimmodel
# Options are: Uhlrich2022 (Raj model update by Uhlrich, 2022). gait2392 will be added shortly
Osim_model = "Uhlrich2022"
model_info_dir = "../Muscle_group_info/"+Osim_model+fsep
Osim_scale_factors = pd.read_csv("../OpenSim_scale_factors/"+Osim_model+".csv")

# Select NN joint model and load necessary info(Muscles are grouped based according to the joints they cross)
# Options are: HipKnee, Hip, Knee, KneeAnkle, Ankle, or Combined. Combined uses model in which all 5 NN are included. The inputs to
# combined are all joint angles and scale factors. The first layer of the Combined model slices the input vector and passes the releveant 
# inputs to each of the sub-models. Finally, the output of each sub-model is concatenated and returned as a prediction.
Model_type = "Combined"
muscles,DOFs,bodies,scale_factors,model_info = get_Muscles_DOFs_Bodies_ScaleFactors(model_info_dir,Model_type)

# Select Muscle paramter type (Models are trained to predict either lengths and moment arms, or lines of action)
# Options are: MA+LMTU or LOA
Param_type = "MA+LMTU"
output_labels = get_output_labels(Model_type,model_info,Param_type,muscles,DOFs)

#%% load desired model and test data. If retrain_model=="yes", training and validation data will also  be loaded
features = DOFs+scale_factors
test_model_dir = MTU_data_path+fsep+Osim_model+fsep+"test"
angles = [s + "_angle" for s in DOFs]
test_inputs, test_labels = get_features_and_labels(test_model_dir,Osim_scale_factors,scale_factors,output_labels,angles)

if (retrain_model==True and use_optimised_model==True) or use_optimised_model==False:
    train_model_dir = MTU_data_path+fsep+Osim_model+fsep+"train"
    val_model_dir = MTU_data_path+fsep+Osim_model+fsep+"validation"
    train_inputs, train_labels = get_features_and_labels(train_model_dir,Osim_scale_factors,scale_factors,output_labels,angles)
    val_inputs, val_labels = get_features_and_labels(val_model_dir,Osim_scale_factors,scale_factors,output_labels,angles)

#%% Load NN model and predict data. If retrain_model=="yes", model weights will be re-initialised and the model will be retrained
#NB: Model training time may be slow if using CPU
b_size = 512
if use_optimised_model==True:
    model = tf.keras.models.load_model(NN_model_path+fsep+Osim_model+fsep+Param_type+fsep+Model_type)
    if retrain_model==True:
        model = tf.keras.models.clone_model(model) # This is a simple method for re-initialising model weights
else:
    # Change functions inputs to modify width, depth, regularisation, and activation functions
    from create_NN import create_NN_model
    model = create_NN_model(input_shape=test_inputs.shape[-1],output_shape=test_labels.shape[-1],
                            n_layers=2,n_nodes=256,activation='swish',L1_penalty=0.001,L2_penalty=0,use_batch_norm=True)
    
if (retrain_model==True and use_optimised_model==True) or use_optimised_model==False:
    output_ranges = np.ptp(train_labels,axis=0)
    loss_weights = list(max(output_ranges)/output_ranges)
    training_callbacks = []
    training_callbacks.append(
        tf.keras.callbacks.EarlyStopping(monitor="val_mae",
                                        patience=50,
                                        min_delta=0.00001))
    training_callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_mae",
                                            factor=0.1,
                                            patience=20,
                                            min_delta=0.00001,
                                            verbose=1,
                                            min_lr=1e-6))
    model.compile(loss='mae',metrics = ['mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),loss_weights=loss_weights)
    model.summary() # Prints a sumamry of model layers
    history = model.fit(train_inputs, train_labels,
            epochs=1000,
            batch_size=b_size,
            validation_data=(val_inputs, val_labels),
            callbacks= training_callbacks)
    
# Predict outputs with model
test_predictions = model.predict(test_inputs,batch_size=b_size)

# %% Organise data for Bland-Altman and Linear regression
test_predictions,test_labels = pd.DataFrame(test_predictions,columns=output_labels), pd.DataFrame(test_labels,columns=output_labels)
NN_all,Osim_all = organise_results(test_predictions,test_labels,muscles,DOFs,output_labels)

# %% Plot results
sparsity =100 # Reduces the number of points in the plot to avoid crashing 
# xticklabels = np.round(np.linspace(np.min(Osim_all['LMTU']),np.max(Osim_all['LMTU']),5),0)

variables = list(NN_all.keys())
ncol = len(variables)
fig,ax = plt.subplots(nrows=2, ncols=ncol, figsize = (ncol*3,5))
ax = ax.flatten()
fig.dpi=300
for i,DOF in enumerate(variables):
    xticks = np.round(np.linspace(np.min(Osim_all[DOF]),np.max(Osim_all[DOF]),5)/5,0)*5
    yticklabels = np.round(np.linspace(np.min(Osim_all[DOF]-NN_all[DOF]),np.max(Osim_all[DOF]-NN_all[DOF]),5)/5,0)*5
    pg.plot_blandaltman(Osim_all[DOF][::sparsity], NN_all[DOF][::sparsity],agreement=1.96,confidence=0.95,xaxis='x',annotate=False,ax=ax[i],s=0.02)
    ax[i] = editBAAnnotation(Osim_all[DOF], NN_all[DOF],ax[i],12)
    ax[i].set_ylim([-4,4])
    ax[i].set_xlabel("")
    ax[i].set_ylabel("")

    ax[i].tick_params(axis='both',direction='out')
    ax[i].set_title(variables[i],fontsize=14)

    ax[i].set_xticks([])

    lm1 = st.linregress(Osim_all[DOF],NN_all[DOF])

    xmin,xmax = np.min(Osim_all[DOF][::sparsity]),np.max(Osim_all[DOF][::sparsity])
    x= np.linspace(xmin,xmax,10)
    y = lm1.slope*x
    r2 = lm1.rvalue**2
    ax[i+ncol].scatter(Osim_all[DOF][::sparsity],NN_all[DOF][::sparsity],s=0.1)
    ax[i+ncol].spines[['right', 'top']].set_visible(False)
    ax[i+ncol].set_xlabel("Osim value (mm)")
    ax[i+ncol].set_ylabel('')
    ax[i+ncol].text(0.05, 0.9, "r2 = "+str(np.round(r2,4)), horizontalalignment='left',
        verticalalignment='center', transform=ax[i+ncol].transAxes)
    ax[i+ncol].set_xticks(xticks)

    if i==0:
        ax[i].set_ylabel('Osim value - NN value')
        ax[i+ncol].set_ylabel('NN value')
fig.align_ylabels()
