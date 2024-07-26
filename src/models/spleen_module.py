import lightning as L
from monai import utils, transforms, networks, data, engines, losses, metrics, visualize, config, inferers, apps
from monai.data import CacheDataset, DataLoader, list_data_collate, pad_list_data_collate, decollate_batch
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import torch
import matplotlib.pyplot as plt
# make sure you have numpy=1.26.0 bc of a bug between newer numpy and monai at the time of this writing. Will surely be solved by monai team in future.
import numpy as np
import pandas as pd
import glob
import os
import shutil
import tempfile
import rootutils


class SpleenLightningModule(L.LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss_fn: torch.nn.Module = None,
        compile: bool = False,
        lr: float = 1e-3,
        ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = networks.nets.UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )

        self.loss_fn = losses.DiceLoss(to_onehot_y=True, softmax=True)
        self.metric = metrics.DiceMetric(include_background=False, reduction="mean")

        # self.post_pred = transforms.Compose([transforms.EnsureType("tensor", device="cpu"), transforms.AsDiscrete(armax=True, to_onehot=2)])
        self.post_pred = transforms.Compose([transforms.EnsureType("tensor", device="cpu"), transforms.AsDiscrete(armax=True)])
        # The post_pred transform is giving me an error that "labels should have a channel with length equal to on" so I think it needs to output a single channel image
        # I think the issue is that the output of the model is a 2 channel image, but the labels are a single channel image
        # I can fix this by adding a channel dimension to the labels
        self.post_label = transforms.Compose([transforms.EnsureType("tensor", device="cpu"), transforms.AsDiscrete(to_onehot=2)])

        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.validation_step_outputs = []
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        # ohhhh. The reason the configs for opt and sched are partial
        # is that they are also passing these other model parameters before instantiating
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self(images)
        loss = self.loss_fn(outputs, labels)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = (160, 160, 160)
        sw_batch_size = 4
        # We should use sliding window inference not because it reduces memory usage (that's not a problem on HPG) but
        # rather because it apparently can increase accuracy by multiple percentage points for things like Dice.
        # We totally could just have outputs = self.forward(images)
        # However, sliding window inference is apparently more accurate for things like Dice.
        outputs = inferers.sliding_window_inference(images, roi_size, sw_batch_size, self.forward)
        loss = self.loss_fn(outputs, labels)
        print("output shape is:")
        print(outputs.shape)
        print("label shape is:")
        print(labels.shape)
        print("decollated batch shape is:")
        print(decollate_batch(outputs)[0].shape)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        dice = self.metric(y_pred=outputs, y=labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/dice", dice.mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        d = {"val_loss": loss, "val_number": len(outputs)}
        self.validation_step_outputs.append(d)
        return d
    
    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.metric.aggregate().item()
        self.metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        self.log("val/mean_dice", mean_val_dice, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/mean_loss", mean_val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.validation_step_outputs.clear()  # free memory
        return
    
    def test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = (160, 160, 160)
        sw_batch_size = 4
        outputs = inferers.sliding_window_inference(images, roi_size, sw_batch_size, self.forward)
        loss = self.loss_fn(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        dice = self.metric(y_pred=outputs, y=labels)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/dice", dice, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return
