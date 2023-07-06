from tensorflow.keras.preprocessing.image import load_img
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
from copy import deepcopy
from tqdm import tqdm


def read_data_by_path(filename):
    """ 
    Read data from data/

    :param filename: the name of directory with data 
    return: name of image files, images samples, and its labels
    """
    txt_file = os.listdir(filename)
    imgfile = []
    label = []

    for line in range(len(txt_file)):
        imgfile.append(filename + txt_file[line].split()[0])
        label.append(int((txt_file[line].split()[0]).split('_')[0]))

    label = np.array(label)-1 

    imgs = np.array([np.array(load_img(im))
        for im in imgfile],'f')

    return np.asarray(imgfile), imgs, label

def save_projection(filename, data, labels, samples):
    """ 
    It saves the tSNE projection of labeled data

    :param filename: filename of the projection to be saved
    :param data: data features
    :param labels: data labels
    :param samples: index of samples to be considered as supervised
    """

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    unsup_samples = np.arange(0, data.shape[0])
    unsup_samples = np.setdiff1d(unsup_samples, samples)
    ax.scatter(data[unsup_samples,0],data[unsup_samples,1],c=labels[unsup_samples],s=20, cmap='tab10', alpha=0.5, edgecolors='none')
    ax.scatter(data[samples,0],data[samples,1],c=labels[samples],s=20, cmap='tab10', edgecolors='red')

    plt.savefig(filename)
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

class iftDataset(Dataset):
    def __init__(self, image_paths, pseudolabels = None, transform=False):
        self.image_paths = image_paths
        self.transform = transform
        self.pseudolabels = pseudolabels
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.pseudolabels is not None:
            label = self.pseudolabels[idx]
        else:
            # ift dataset has labels from 1-n, and models consider label 0.
            label = int(image_filepath.split('/')[-1].split('_')[0])  
            label = label - 1

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label


@torch.no_grad()
def prepare_data_features(model, dataset, device, num_workers):
    # Prepare model
    network = deepcopy(model.convnet)
    #if layer=='backbone':
    #    network.fc = nn.Identity()  # Removing projection head g(.)

    network.eval()
    network.to(device)

    # Encode all images
    data_loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=False, drop_last=False)
    feats, labels = [], []
    for batch_imgs, batch_labels in tqdm(data_loader):
        batch_imgs = batch_imgs.to(device)
        batch_feats = network(batch_imgs)
        feats.append(batch_feats.detach().cpu())
        labels.append(batch_labels)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)

    return TensorDataset(feats, labels), feats.numpy(), labels.numpy()

def make_feats_as_tensor(feats, labels):
    feats = torch.from_numpy(feats)
    labels = torch.from_numpy(labels)
    labels = labels.type(torch.LongTensor)
    return data.TensorDataset(feats, labels)
