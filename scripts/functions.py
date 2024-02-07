
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