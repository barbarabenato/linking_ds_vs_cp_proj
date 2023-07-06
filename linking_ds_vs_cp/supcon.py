import os
from tqdm import tqdm
from copy import deepcopy
import utils as u

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models

import pytorch_lightning as pl
from pytorch_metric_learning import losses

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


class SupCon(pl.LightningModule):
    def __init__(self, hidden_dim, num_classes, lr, temperature, weight_decay, device, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"

        # Base model f(.)
        self.convnet = models.resnet18(
            pretrained=True
        )  # num_classes is the output size of the last linear layer
        self.convnet.fc = nn.Linear(512, 4 * hidden_dim)
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.dev = device
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def supcon_loss(self, batch, mode="train"):
        imgs, labels = batch
        imgs = torch.cat(imgs, dim=0)
        batch_size = len(labels)

        # Encode all images
        feats = self.convnet(imgs)
        f1, f2 = torch.split(feats, [batch_size, batch_size], dim=0)
        feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) 
        labels = torch.cat([labels.unsqueeze(1), labels.unsqueeze(1)], dim=1)
        
        feats = torch.cat(torch.unbind(feats, dim=1), dim=0)
        labels = torch.cat(torch.unbind(labels, dim=1), dim=0)
        labels = labels.float().to(self.dev)

        # loss
        loss_fn = losses.SupConLoss(temperature=0.07)
        loss = loss_fn(feats, labels)
        # Logging loss
        self.log(mode + "_loss", loss)

        return loss

    def training_step(self, batch, batch_idx):
        return self.supcon_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.supcon_loss(batch, mode="val")

    # Avoiding problems with dimension mismatch when loading the model after
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                is_changed = True
        if is_changed:
            checkpoint.pop("optimizer_states", None)
