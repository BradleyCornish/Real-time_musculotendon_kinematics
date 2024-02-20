
import numpy as np
import pandas as pd
from matplotlib import transforms
from os import listdir
fsep ="/"
def editBAAnnotation(x,y,a1,FSize):
    # inputs are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    n = x.size 

    diff = x- y
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    mean_diff_se = np.sqrt(std_diff**2/n)

    agreement = 1.96

    high = mean_diff + (agreement * std_diff)
    low = mean_diff - (agreement * std_diff)
    high_low_se = np.sqrt(3 * std_diff**2 / n)
    
    # annotations to be modified in the graph
    if True:
        loa_range = high - low
        offset1 = (loa_range / 30.0) * 1.5
        offset2 = 0.8
        trans = transforms.blended_transform_factory(
            a1.transAxes, a1.transData)
        xloc = 0.98
        # a1.text(xloc, mean_diff + offset1, 'Mean', ha="right", va="bottom",

        #         transform=trans)
        # a1.text(xloc, mean_diff - offset1, '%.2f' % mean_diff, ha="right",
        #         va="top", transform=trans)
        # a1.text(xloc, high + offset, '+%.2f SD' % agreement, ha="right",
        #        va="bottom", transform=trans)
        a1.text(xloc, high + offset1, '%.2f' % high, ha="right", va="bottom",
                transform=trans,fontsize=FSize)
        # a1.text(xloc, low - offset, '-%.2f SD' % agreement, ha="right",
        #             va="top", transform=trans)
        a1.text(xloc, low - offset1, '%.2f' % low, ha="right", va="top",
                transform=trans,fontsize=FSize)
    
    return a1

def get_features_and_labels(model_dir,Osim_scale_factors,scale_factors_of_interest,output_labels,angles):
    features, labels =[],[]
    models = listdir(model_dir)
    for t_model in models:
        model_idx = int(str.removeprefix(str.removesuffix(t_model,'.csv'),'model_'))
        scales = Osim_scale_factors.loc[Osim_scale_factors['Model'] == model_idx,scale_factors_of_interest].values
        data = pd.read_csv(model_dir+fsep+t_model)
        model_features = np.hstack([data.loc[:,angles].values,np.repeat(scales,data.shape[0],axis=0)])
        model_labels = data.loc[:,output_labels].values
        features.append(model_features)
        labels.append(model_labels)
    features, labels= np.vstack(features), np.vstack(labels)
    return features, labels


def get_output_labels(Model_type,model_info,Param_type,muscles,DOFs):
    if Model_type == "Combined":
        Model_list = ["HipKnee","Hip","Knee","KneeAnkle","Ankle"]
        output_labels=[]   
        for Model in Model_list:
            muscles_in_model = list(model_info['muscles'].loc[:,Model].dropna().values)
            DOFs_in_model = list(model_info['DOFs'].loc[:,Model].dropna().values)

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
    return output_labels

def organise_results(test_predictions,test_labels,muscles,DOFs,output_labels):
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
    return NN_all,Osim_all

def get_Muscles_DOFs_Bodies_ScaleFactors(model_info_dir, Model_type):
    model_info={}
    model_info['muscles'] = pd.read_csv(model_info_dir+"/"+"Muscles.csv")
    model_info['DOFs'] = pd.read_csv(model_info_dir+"/"+"DegreesOfFreedom.csv")
    model_info['bodies'] = pd.read_csv(model_info_dir+"/"+"ForcesOnBodies.csv")
    model_info['scale_factors'] = pd.read_csv(model_info_dir+"/"+"ScaleFactorsOfInterest.csv")

    muscles = list(model_info['muscles'] .loc[:,Model_type].dropna().values)
    DOFs = list(model_info['DOFs'].loc[:,Model_type].dropna().values)
    bodies = list(model_info['bodies'].loc[:,Model_type].dropna().values)
    scale_factors = list(model_info['scale_factors'].loc[:,Model_type].dropna().values)
    return muscles,DOFs,bodies,scale_factors,model_info



