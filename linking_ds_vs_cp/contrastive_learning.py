import os
import utils as u
import contrastive_learning_models as cl_models

import torch
from torchvision import transforms
import pytorch_lightning as pl


class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]
    
contrast_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=96),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

img_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def contr_learning(model_name, sup_paths, unsup_paths, n_classes, epochs):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("[data] Loading data")
    # Loading data for generating views in contrastive learning with transformations
    n_views = 2
    sup_dataset = u.iftDataset(sup_paths,None, ContrastiveTransformations(contrast_transforms, n_views=n_views))
    unsup_dataset = u.iftDataset(unsup_paths,None, ContrastiveTransformations(contrast_transforms, n_views=n_views)) #test transforms are applied

    sup_loader = u.DataLoader(sup_dataset, batch_size=8, shuffle=True,num_workers=1)
    unsup_loader = u.DataLoader(unsup_dataset, batch_size=8, shuffle=True,num_workers=1)

    if model_name == 'simclr': 
        print("[training] Training SimCLR")
        # Training SimCLR
        model = cl_models.train_simclr(
            sup_loader,
            unsup_loader,
            device,
            batch_size=8, 
            hidden_dim=1024, 
            num_classes = n_classes, 
            lr=5e-4, 
            temperature=0.07,
            weight_decay=1e-4,
            max_epochs=epochs
        )
    elif model_name == 'supcon':
        print("[training] Training SupCon")
        # Training SupCon
        model = cl_models.train_supcon(
            sup_loader,
            unsup_loader,
            device,
            batch_size=128, 
            hidden_dim=1024,
            num_classes = n_classes, 
            lr=5e-4, 
            temperature=0.07,
            weight_decay=1e-4, 
            max_epochs=epochs
        )
    else:
        print("[training] Training SimCLR")
        # Training SimCLR
        simclr_model = cl_models.train_simclr(
            sup_loader,
            unsup_loader,
            device,
            batch_size=8, 
            hidden_dim=1024,
            num_classes = n_classes, 
            lr=5e-4,
            temperature=0.07,
            weight_decay=1e-4, 
            max_epochs=epochs
        )
        print("[training] Finetuning SimCLR with SupCon")
        # Finetuning with SupCon
        model = cl_models.train_supcon_from_simclr(
            simclr_model,
            sup_loader,
            unsup_loader,
            device,
            batch_size=128, 
            hidden_dim=1024,
            num_classes = n_classes, 
            lr=5e-4, 
            temperature=0.07,
            weight_decay=1e-4, 
            max_epochs=epochs
        )

    # Loading data to extracti features without contrastive transformations
    sup_img_data = u.iftDataset(sup_paths, None, img_transforms)
    unsup_img_data = u.iftDataset(unsup_paths, None, img_transforms)

    # Extracting features and training decision layer
    print("[extracting] Extracting features for model ")
    # getting features from original data (without data augmentation) 
    _, train_feats, train_labs = u.prepare_data_features(model, sup_img_data, device, 1)
    _, unsup_feats, unsup_labs = u.prepare_data_features(model, unsup_img_data, device, 1)

    return train_feats, train_labs, unsup_feats, unsup_labs


