import os
import utils as u
from simclr import SimCLR
from supcon import SupCon
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

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

def train_simclr(sup_loader, unsup_loader, device, batch_size, max_epochs=500, **kwargs):
    trainer = pl.Trainer(
        gpus=1 if str(device) == "cuda" else 0,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc_top5"),
            LearningRateMonitor("epoch"),
        ],
        progress_bar_refresh_rate=1,
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
    pl.seed_everything(42)  # To be reproduciable
    model = SimCLR(max_epochs=max_epochs, **kwargs)
    trainer.fit(model, unsup_loader, sup_loader)
    # Load best checkpoint after training
    model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    return model


def train_supcon_from_simclr(model_unsup, sup_loader, unsup_loader, device, batch_size, max_epochs=500, **kwargs):
    trainer = pl.Trainer(
        gpus=1 if str(device) == "cuda" else 0,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="train_loss"),
            LearningRateMonitor("epoch"),
        ],
        progress_bar_refresh_rate=1,
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
    pl.seed_everything(42)  # To be reproduciable
    model = SupCon(device = device, max_epochs=max_epochs, **kwargs)
    model.load_state_dict(model_unsup.state_dict())
    trainer.fit(model, sup_loader) 
    return model


def train_supcon(sup_loader, unsup_loader, device, batch_size, max_epochs=500, **kwargs):
    trainer = pl.Trainer(
        gpus=1 if str(device) == "cuda" else 0,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
            LearningRateMonitor("epoch"),
        ],
        progress_bar_refresh_rate=1,
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
    pl.seed_everything(42)  # To be reproducable
    model = SupCon(device = device, max_epochs=max_epochs, **kwargs)
    trainer.fit(model, sup_loader) 
    return model


def learning_feature_space(model_name, sup_paths, unsup_paths, n_classes, epochs):
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
        model = train_simclr(
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
        model = train_supcon(
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
        simclr_model = train_simclr(
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
        model = train_supcon_from_simclr(
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

    # Creating data loader for original data (without contrastive transformations)
    sup_img_data = u.iftDataset(sup_paths, None, img_transforms)
    unsup_img_data = u.iftDataset(unsup_paths, None, img_transforms)

    # Extracting features 
    print("[extracting] Extracting features for model ")
    # getting features from original data  
    _, train_feats, train_labs = u.prepare_data_features(model, sup_img_data, device, 1)
    _, unsup_feats, unsup_labs = u.prepare_data_features(model, unsup_img_data, device, 1)

    return train_feats, train_labs, unsup_feats, unsup_labs



