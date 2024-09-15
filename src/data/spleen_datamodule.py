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


class SpleenDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        root_dir: str,
        split_name: str,    # name of the split
        image_src: str,     # absolute directory input images
        label_src: str,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        # the above line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt

        # We first need to define the transforms that we will apply to the data
        # Transforms may sound like they just augment the data, but they can also be used to preprocess the data, including loading the data and converting it to the correct format
        self.train_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                # transforms.Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                transforms.Resized(keys=["image", "label"], spatial_size=(96, 96, 96), mode=("bilinear", "nearest")),
                transforms.SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96), method="symmetric"),
                transforms.ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
                # transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                # transforms.EnsureTyped(keys=["image", "label"]),
                # transforms.RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=4),
                transforms.RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(64, 64, 64), pos=1, neg=1, num_samples=4),
            ]
        )

        self.val_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                # transforms.Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                transforms.Resized(keys=["image", "label"], spatial_size=(96, 96, 96), mode=("bilinear", "nearest")),
                transforms.SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96), method="symmetric"),
                transforms.ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                # transforms.EnsureTyped(keys=["image", "label"]),
            ]
        )
        
        self.test_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                # transforms.Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                transforms.Resized(keys=["image", "label"], spatial_size=(96, 96, 96), mode=("bilinear", "nearest")),
                transforms.SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96), method="symmetric"),
                transforms.ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                # transforms.EnsureTyped(keys=["image", "label"]),
            ]
        )

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        # we already downloaded our data so we will not do anything here
        pass

    def setup(self, stage=None):
        # the stage is 'fit', 'validate', 'test', or 'predict'
        # Here is where we make assignments here (val/train/test split)
        # called on every process in DDP

        # TODO: os.path.join() is not working???
        # INFO: Apparently, os.path.join() should not have a '/' at the beginning of the second argument. It should be like os.path.join(root_dir, "data/Task09_Spleen/imagesTr")
        IMAGE_SRC = os.path.join(self.hparams.root_dir,"data/Task09_Spleen/imagesTr")
        # IMAGE_SRC = str(root_dir) + "/data/Task09_Spleen/imagesTr"
        LABEL_SRC = os.path.join(self.hparams.root_dir,"data/Task09_Spleen/labelsTr")
        # LABEL_SRC = str(root_dir) + "/data/Task09_Spleen/labelsTr"
        # FIX: Pass SPLITS_DIR as well as SPLIT_NAME
        # Maybe I should just pass the cfg.paths dictionary.
        # What about the SPLIT_NAME though?
        # I think SPLITS_DIR should just be the /splits/ directory. Need to pass root dir somehow though.
        SPLIT_NAME = "MySplit"
        # this can be done by stage
        # filenames = None
        # TODO: Use a cute little dictionary to map fit/validate/test terms to train/val/test for the below

        # if stage == "fit":
        #     filenames = pd.read_csv(f"/splits/{SPLIT_NAME}/train_{SPLIT_NAME}.csv")
        #     # Create a dictionary list of the image and label files labelled as 'image' and 'label'
        #     self.train_files = [{"image": os.path.join(IMAGE_SRC, f"{filename}.nii.gz"), "label": os.path.join(LABEL_SRC, f"{filename}.nii.gz")} for filename in filenames]
        #     self.train_ds = CacheDataset(data=self.train_files, transform=self.train_transforms, cache_rate=1.0, num_workers=self.num_workers)
        # elif stage == "validate":
        #     filenames = pd.read_csv(f"/splits/{SPLIT_NAME}/val_{SPLIT_NAME}.csv")
        #     self.val_files = [{"image": os.path.join(IMAGE_SRC, f"{filename}.nii.gz"), "label": os.path.join(LABEL_SRC, f"{filename}.nii.gz")} for filename in filenames]
        #     self.val_ds = CacheDataset(data=self.val_files, transform=self.val_transforms, cache_rate=1.0, num_workers=self.num_workers)
        # elif stage == "test":
        #     filenames = pd.read_csv(f"/splits/{SPLIT_NAME}/test_{SPLIT_NAME}.csv")
        #     self.test_files = [{"image": os.path.join(IMAGE_SRC, f"{filename}.nii.gz"), "label": os.path.join(LABEL_SRC, f"{filename}.nii.gz")} for filename in filenames]
        #     self.test_ds = CacheDataset(data=self.test_files, transform=self.test_transforms, cache_rate=1.0, num_workers=self.num_workers)
        # else:
        #     raise ValueError(f"Stage {stage} not supported")

        # TODO: os.path.join() is not working for some reason
        print("root dir is:")
        print(self.hparams.root_dir)
        train_csv = os.path.join(str(self.hparams.root_dir),f"splits/{SPLIT_NAME}/train_{SPLIT_NAME}.csv")
        # train_csv = f"/splits/{SPLIT_NAME}/train_{SPLIT_NAME}.csv"
        # train_csv = os.path.join(str(root), train_csv)
        # train_csv = str(root) + train_csv
        train_filenames = pd.read_csv(train_csv)
        self.train_files = [{"image": os.path.join(IMAGE_SRC, row["image"]), "label": os.path.join(LABEL_SRC, row["label"])} for index, row in train_filenames.iterrows()]
        # Take only first few files of the dataset for testing
        self.train_files = self.train_files[:2]     # FIX: take only the first 5 files for testing; COMMENT THIS LINE OUT FOR ACTUAL TRAINING
        self.train_ds = CacheDataset(data=self.train_files, transform=self.train_transforms, cache_rate=1.0, num_workers=self.hparams.num_workers)
        
        # val_csv = f"/splits/{SPLIT_NAME}/val_{SPLIT_NAME}.csv"
        # val_csv = str(root) + val_csv
        val_csv = os.path.join(str(self.hparams.root_dir),f"splits/{SPLIT_NAME}/val_{SPLIT_NAME}.csv")
        val_filenames = pd.read_csv(val_csv)
        self.val_files = [{"image": os.path.join(IMAGE_SRC, row["image"]), "label": os.path.join(LABEL_SRC, row["label"])} for index, row in val_filenames.iterrows()]
        self.val_files = self.val_files[:2]     # FIX: take only the first 5 files for testing; COMMENT THIS LINE OUT FOR ACTUAL TRAINING
        self.val_ds = CacheDataset(data=self.val_files, transform=self.val_transforms, cache_rate=1.0, num_workers=self.hparams.num_workers)
        
        # test_csv = f"/splits/{SPLIT_NAME}/test_{SPLIT_NAME}.csv"
        # test_csv = str(root) + test_csv
        test_csv = os.path.join(str(self.hparams.root_dir),f"splits/{SPLIT_NAME}/test_{SPLIT_NAME}.csv")
        test_filenames = pd.read_csv(test_csv)
        self.test_files = [{"image": os.path.join(IMAGE_SRC, row["image"]), "label": os.path.join(LABEL_SRC, row["label"])} for index, row in test_filenames.iterrows()]
        self.test_files = self.test_files[:2]     # FIX: take only the first 5 files for testing; COMMENT THIS LINE OUT FOR ACTUAL TRAINING
        self.test_ds = CacheDataset(data=self.test_files, transform=self.test_transforms, cache_rate=1.0, num_workers=self.hparams.num_workers)
        


    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.hparams.batch_size,         # we can use this nifty trick and access the hyperparameters directly since we used self.save_hyperparameters() up top
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=list_data_collate,               # this collates our list of dictionaries into a dictionary of lists; not needed for if your dataset outputs something the default collate_fn can handle
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=pad_list_data_collate,
            shuffle=False,
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=list_data_collate,
            shuffle=False,
        )
