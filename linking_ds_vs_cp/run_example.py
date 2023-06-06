from utils import read_data_by_path
from utils import save_projection
from feature_learning import vgg16_learning
from contrastive_learning import contr_learning
from dim_red import dim_red
from label_propag import OPFSemi

import numpy as np
import os
import sys 

if len(sys.argv) != 5:
    sys.exit('\tusage run_example.py <model name [simclr, supcon, both]> <perc of sup samples> <dimensionality reduction> <iterations>')

model_name = sys.argv[1]
perc_sup_samples = float(sys.argv[2])
dim_red_name = sys.argv[3]
epochs = int(sys.argv[4])

# params for vgg_learning batch size, number of epochs, and number of classes
feat_learn_params = [32, 15, 10]

# reading data
imgfile, x, y = read_data_by_path('../BaseParasitos/parasites_focus_plane_divided/larvas/resized/')

# randomly choosing the supervised samples
idx_sup_samples = np.random.randint(0, high=x.shape[0], size=(int(x.shape[0]*perc_sup_samples),))
idx_unsup_samples = np.diff(np.arange(0, high=x.shape[0]), idx_sup_samples)

# training deep feature learning with supervised samples in the first iteration
print("[] contrastive learning")
sup_X, sup_y, unsup_X, unsup_y = contr_learning(model_name, imgfile[idx_sup_samples], imgfile[idx_unsup_samples], feat_learn_params[2], epochs)

# dimensionality reduction step 
print("[] computing 2d projection")
sup_X, unsup_X = dim_red(dim_red_name, sup_X, sup_y, unsup_X, unsup_y)
feats_2d = np.concatenate((sup_X,unsup_X))

# propagating labels with OPFSemi
print("[] propagating labels")
y_labeled = OPFSemi(feats_2d, y, idx_sup_samples)

save_projection('output/2d_data.png', feats_2d, y_labeled, idx_sup_samples)

# learning vgg16 
feats_nd = vgg16_learning(x, y_labeled, idx_sup_samples, batch=feat_learn_params[0], epochs=feat_learn_params[1], file_name='learning_curve_iter.png', n_classes=feat_learn_params[2])


