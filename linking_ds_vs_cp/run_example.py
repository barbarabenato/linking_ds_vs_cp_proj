from utils import read_data_by_path
from utils import save_projection
from classifier_learning import vgg16_classification
from feature_learning import learning_feature_space
from dimensionality_reduction import reduce_to_2d
from label_propagation import OPFSemi
from sklearn.metrics import cohen_kappa_score, accuracy_score

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
class_learn_params = [32, 15, 10]

# reading data
imgfile, original_img, _ = read_data_by_path('../data/')

# randomly choosing the supervised samples
idx_sup_samples = np.random.choice(range(len(imgfile)), int(len(imgfile)*perc_sup_samples), replace=False)
idx_unsup_samples = np.setdiff1d(np.arange(0, len(imgfile)),idx_sup_samples)

# training deep feature learning with supervised samples 
print("[] feature learning")
X_sup, y_sup, X_unsup, y_unsup = learning_feature_space(model_name, imgfile[idx_sup_samples], imgfile[idx_unsup_samples], class_learn_params[2], epochs)

# dimensionality reduction step 
print("[] computing 2d projection")
feats_nd = np.concatenate((X_sup, X_unsup))
feats_2d = reduce_to_2d(dim_red_name, feats_nd)

# propagating labels with OPFSemi
print("[] propagating labels")
y = np.concatenate((y_sup, y_unsup))

y_labeled = OPFSemi(feats_2d, y, idx_sup_samples)
save_projection('2d_data.png', feats_2d, y_labeled, idx_sup_samples)

# learning vgg16 
y_classified = vgg16_classification(original_img, y_labeled, class_learn_params, file_name='learning_curve.png')

prop_acc = accuracy_score(y, y_labeled)
prop_kappa = cohen_kappa_score(y, y_labeled)
print("Propagation acc: %f\nPropagation kappa: %f" % (prop_acc, prop_kappa))

acc = accuracy_score(y, y_classified)
kappa = cohen_kappa_score(y, y_classified)
print("Classification acc: %f\nClassification kappa: %f" % (acc, kappa))

