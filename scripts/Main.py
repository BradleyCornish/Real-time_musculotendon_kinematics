#%%
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pingouin as pg
from scipy import stats as st
# import random
# from math import pi, sqrt, cos, sin
# from datetime import date
from functions import editBAAnnotation, get_features_and_labels

fsep="/"
retrain_model ="no"
NN_model_path = "../NN_models"
MTU_data_path = "../MTU_data"

# Select OpenSimmodel
# Options are: Uhlrich2022 (Raj model update by Uhlrich, 2022), gait2392.
Osim_model = "Uhlrich2022"
model_info = pd.read_excel("../OsimModelInfo/"+Osim_model+"_info.xlsx",sheet_name=["Muscles","DegreesOfFreedom","ScaleFactors","ForcesOnBodies"])
Osim_scale_factors = pd.read_csv("../Osim_scale_factors/"+Osim_model+".csv")
# Select NN joint model and load necessary info(Muscles are grouped based according to the joints they cross)
# Options are: HipKnee, Hip, Knee, KneeAnkle, Ankle, or Combined. Combined uses model in which all 5 NN are included. The inputs to
# combined are all joint angles and scale factors. The first layer of the Combined model slices the input vector and passes the releveant 
# inputs to each of the sub-models. Finally, the output of each sub-model is concatenated and returned as a prediction.
Model_type = "Combined"
muscles = list(model_info['Muscles'].loc[:,Model_type].dropna().values)
DOFs = list(model_info['DegreesOfFreedom'].loc[:,Model_type].dropna().values)
bodies = list(model_info['ForcesOnBodies'].loc[:,Model_type].dropna().values)
scale_factors = list(model_info['ScaleFactors'].loc[:,Model_type].dropna().values)

# Select Muscle paramter type (Models are trained to predict either lengths and moment arms, or lines of action)
# Options are: MA+LMTU or LOA
Param_type = "MA+LMTU"

if Model_type == "Combined":
    Model_list = ["HipKnee","Hip","Knee","KneeAnkle","Ankle"]
    output_labels=[]   
    for Model in Model_list:
        muscles_in_model = list(model_info['Muscles'].loc[:,Model].dropna().values)
        DOFs_in_model = list(model_info['DegreesOfFreedom'].loc[:,Model].dropna().values)

        if Param_type=="MA+LMTU":
            label_variables = ["Ma" + s for s in DOFs_in_model]
            label_variables = ["length"] + label_variables
        elif Param_type=="LOA":
            label_variables = ["Fx","Fy","Fz"]

        if Param_type=="MA+LMTU":
            LMTU_labels, MA_labels=[],[]
            for muscle in muscles_in_model:
                LMTU_labels.append(muscle+"_r_length")
                for DOF in ["Ma" + s for s in DOFs_in_model]:
                    MA_labels.append(muscle+"_r_"+DOF)
            output_labels = output_labels+LMTU_labels+MA_labels
        elif Param_type=="LOA":
            for muscle in muscles_in_model:
                for direction in ["Fx","Fy","Fz"]:
                    output_labels.append(muscle+"_r_"+direction)

else:
    if Param_type=="MA+LMTU":
        label_variables = ["Ma" + s for s in DOFs]
        label_variables = ["length"] + label_variables
    elif Param_type=="LOA":
        label_variables = ["Fx","Fy","Fz"]

    output_labels=[]    
    if Param_type=="MA+LMTU":
        LMTU_labels, MA_labels=[],[]
        for muscle in muscles:
            LMTU_labels.append(muscle+"_r_length")
            for DOF in ["Ma" + s for s in DOFs]:
                MA_labels.append(muscle+"_r_"+DOF)
        output_labels = LMTU_labels+MA_labels
    elif Param_type=="LOA":
        for muscle in muscles:
            for direction in ["Fx","Fy","Fz"]:
                output_labels.append(muscle+"_r_"+direction)

#%% load desired model and test data. If retrain_model=="yes", training and validation data will also  be loaded
features = DOFs+scale_factors
test_model_dir = MTU_data_path+fsep+Osim_model+fsep+"test"
angles = [s + "_angle" for s in DOFs]

test_inputs, test_labels = get_features_and_labels(test_model_dir,Osim_scale_factors,scale_factors,output_labels,angles)
if retrain_model=="yes":
    train_model_dir = MTU_data_path+fsep+Osim_model+fsep+"train"
    val_model_dir = MTU_data_path+fsep+Osim_model+fsep+"validation"
    train_inputs, train_labels = get_features_and_labels(train_model_dir,Osim_scale_factors,scale_factors,output_labels,angles)
    val_inputs, val_labels = get_features_and_labels(val_model_dir,Osim_scale_factors,scale_factors,output_labels,angles)
#%% Load NN model and predict data. If retrain_model=="yes", model weights will be re-initialised and the model will be retrained
#NB: Model training time may be slow if using CPU
model = tf.keras.models.load_model(NN_model_path+fsep+Osim_model+fsep+Param_type+fsep+Model_type)
b_size = 512
if retrain_model=="yes":
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

    model = tf.keras.models.clone_model(model) # This is a simple method for re-initialising model weights
    model.compile(loss='mae',metrics = ['mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),loss_weights=loss_weights)
    model.summary() # Prints a sumamry of model layers

    history = model.fit(train_inputs, train_labels,
            epochs=1000,
            batch_size=b_size,
            validation_data=(val_inputs, val_labels),
            callbacks= training_callbacks)
test_predictions = model.predict(test_inputs,batch_size=b_size)

# %% Setup data for Bland-Altman and Linear regression
test_predictions,test_labels = pd.DataFrame(test_predictions,columns=output_labels), pd.DataFrame(test_labels,columns=output_labels)
NN,Osim,Residuals,R2 = {},{},{},{}


variables = ['LMTU','MA']
NN_all,Osim_all ={},{}
for variable in variables:
    NN[variable],Osim[variable],Residuals[variable],R2[variable] = {},{},{},{}
    if variable=='MA':
        for DOF in ["Ma" + s for s in DOFs]:
            NN_all[DOF],Osim_all[DOF] = [],[]
            NN[variable][DOF],Osim[variable][DOF],Residuals[variable][DOF],R2[variable][DOF] = {},{},{},{}
    else:
        NN_all[variable],Osim_all[variable] = [],[]

for muscle in muscles:
    # idx=[index for index, value in enumerate(output_labels) if 'length' in value and muscle in value][0]
    NN['LMTU'][muscle]=test_predictions.loc[:,muscle+'_r_length'].values
    Osim['LMTU'][muscle]=test_labels.loc[:,muscle+'_r_length'].values
    NN_all['LMTU'].append(NN['LMTU'][muscle])
    Osim_all['LMTU'].append(Osim['LMTU'][muscle])
    # Residuals['LMTU'][muscle] = Osim['LMTU'][muscle] - NN['LMTU'][muscle]
    # R2['LMTU'][muscle] = st.pearsonr(Osim['LMTU'][muscle], NN['LMTU'][muscle])

    for DOF in ["Ma" + s for s in DOFs]:
        if muscle+'_r_'+DOF in output_labels:
            NN['MA'][DOF][muscle] = test_predictions.loc[:,muscle+'_r_'+DOF].values
            Osim['MA'][DOF][muscle] = test_labels.loc[:,muscle+'_r_'+DOF].values
            # Residuals['MA'][DOF][muscle] = Osim['MA'][DOF][muscle] - NN['MA'][DOF][muscle]
            NN_all[DOF].append(NN['MA'][DOF][muscle])
            Osim_all[DOF].append(Osim['MA'][DOF][muscle])
for key in NN_all.keys():
    NN_all[key]=np.hstack(NN_all[key])*1000 # Convert to mm
    Osim_all[key]=np.hstack(Osim_all[key])*1000 # Convert to mm
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
    # lims = ax[i].get_ylim()
    # ax[i].set_ylim([-1*max(np.abs(lims)),max(np.abs(lims))])
    # yticks = np.round(np.linspace(-1*max(np.abs(lims)),max(np.abs(lims)),5),1)
    # ax[i].set_yticks(yticks)
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
        

# %%
