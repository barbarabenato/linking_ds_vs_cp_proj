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
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_metric_learning import losses

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


class SimCLR(pl.LightningModule):
    def __init__(self, hidden_dim, num_classes, lr, temperature, weight_decay, arch, max_epochs=500):
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
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode="train"):
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)

        # Encode all images
        feats = self.convnet(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")

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


class SupCon(pl.LightningModule):
    def __init__(self, hidden_dim, num_classes, lr, temperature, weight_decay, device, arch, max_epochs=500):
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
    pl.seed_everything(42)  # To be reproducable
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
    pl.seed_everything(42)  # To be reproducable
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

