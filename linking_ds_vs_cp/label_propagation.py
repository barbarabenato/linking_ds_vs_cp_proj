import pyift.pyift as ift
import sys
import numpy as np


def OPFSemi(x, y, samples):
    """ 
    It creates iftdataset format and calls OPFSemi.

    :param x: data features
    :param y: data labels
    :param samples: index of samples to be considered as supervised
    Samples with threshold up to the confidence will be selected to retrain the feature
    learning in the next iteration. To select all samples, conf_threshold=0.0
    return: pseudolabels and confidence values
    """
    # creating opfdataset with provided features
    Z = ift.CreateDataSetFromNumPy(x, np.array(y+1, dtype="int32")) # opf dataset considers labels as [1,n]
    # defining status as training (4) for all samples
    stat = np.full((x.shape[0],), fill_value=(4), dtype="uint32")
    # defining status as supervised (68) only for supervised samples
    stat[samples] = 68

    # setting information to opf dataset
    Z.SetStatus(np.array(stat, dtype="int32"))
    Z.SetTrueLabels(np.array(y+1, dtype="int32"))
    Z.SetNTrainSamples(x.shape[0])

    # creating semi-supervised graph
    graph = ift.SemiSupTrain(Z)
    labels = Z.GetLabels()-1 # returning labels as [0,n-1]
    
    return labels

