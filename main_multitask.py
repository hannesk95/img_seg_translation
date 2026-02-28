from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    AsDiscreted,
    NormalizeIntensityd,
    RandCropByLabelClassesd,
    EnsureTyped,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch, Dataset, PersistentDataset
from monai.apps import download_and_extract
from monai.transforms import MapTransform

import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter
import tempfile
import os
import glob
import mlflow
from monai.handlers.utils import from_engine
from utils import set_deterministic, log_all_python_files
from itertools import cycle, islice
from monai.metrics import HausdorffDistanceMetric

# ---------------------------
# Custom Transform: 3D Features
# ---------------------------
class Compute3DFeaturesd(MapTransform):
    def __init__(self, keys, sigma=1.0):
        super().__init__(keys)
        self.sigma = sigma

    def __call__(self, data):
        d = dict(data)
        img = d["image"][0]  # remove channel for scipy

        Dx = gaussian_filter(img, sigma=self.sigma, order=[1, 0, 0])
        Dy = gaussian_filter(img, sigma=self.sigma, order=[0, 1, 0])
        Dz = gaussian_filter(img, sigma=self.sigma, order=[0, 0, 1])
        grad_mag = np.sqrt(Dx**2 + Dy**2 + Dz**2)

        features = np.stack([Dx, Dy, Dz, grad_mag], axis=0)

        # normalize for stability
        features = features / (np.std(features) + 1e-6)

        d["features"] = features.astype(np.float32)
        return d


# ---------------------------
# Multi-Head UNet
# ---------------------------
class MultiTaskUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()

        self.backbone = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=16,  # shared representation
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.INSTANCE,
        )

        self.seg_head = nn.Conv3d(16, out_channels, kernel_size=1)
        self.feat_head = nn.Conv3d(16, 4, kernel_size=1)  # Dx, Dy, Dz, |grad|

    def forward(self, x):
        shared = self.backbone(x)
        seg_out = self.seg_head(shared)
        feat_out = self.feat_head(shared)
        return seg_out, feat_out

EPOCHS = 1000
ITERATIONS = 100

def main(task: str, n_classes: int, patch_size: tuple, n_channels: int):

    torch.cuda.empty_cache()
    set_determinism(seed=0)
    set_deterministic()
    log_all_python_files()    

    mlflow.log_param("task", task)
    mlflow.log_param("n_classes", n_classes)
    mlflow.log_param("train_setup", "multitask_segmentation")

    directory = "./data"
    data_dir = os.path.join(directory, task)

    train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

    # split data dicts into train and validation sets using 80/20 ratio
    split_idx = int(0.8 * len(data_dicts))
    train_files, val_files = data_dicts[:split_idx], data_dicts[split_idx:]   

    scaling = NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True) if task != "Task06_Lung" else ScaleIntensityRanged(
                keys=["image"],
                a_min=-1000,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest")),
            scaling,

            # FIRST crop
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                num_classes=class_dict[task],
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),

            # THEN compute features on patch
            Compute3DFeaturesd(keys=["image"]),

            EnsureTyped(keys=["image", "label", "features"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest")),
            scaling,

            # FIRST crop
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                num_classes=class_dict[task],
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),

            # THEN compute features on patch
            Compute3DFeaturesd(keys=["image"]),

            EnsureTyped(keys=["image", "label", "features"]),
        ]
    )

    # train_ds = CacheDataset(train_files, train_transforms, cache_rate=0.0, num_workers=4)
    train_ds = PersistentDataset(train_files, train_transforms, cache_dir="./data/cache_multitask")
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

    # val_ds = CacheDataset(val_files, val_transforms, cache_rate=0.0, num_workers=4)
    val_ds = PersistentDataset(val_files, val_transforms, cache_dir="./data/cache_multitask")
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    # ---------------------------
    # Model & Loss
    # ---------------------------
    device = torch.device("cuda:0")
    model = MultiTaskUNet(in_channels=n_channels, out_channels=n_classes).to(device)

    dice_loss = DiceCELoss(to_onehot_y=True, softmax=True)
    feat_loss = nn.L1Loss()

    optimizer = torch.optim.SGD(model.parameters(), 1e-2, momentum=0.99, nesterov=True, weight_decay=3e-5)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=EPOCHS, power=0.9)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hausdorff_metric = HausdorffDistanceMetric(
        include_background=False,
        reduction="mean",
        percentile=95.0,   # HD95 (recommended)
        get_not_nans=True  # avoids NaN if class missing
    )

    lambda_feat = 0.05  # tune this carefully

    val_interval = 2
    best_metric = -1

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=n_classes)])
    post_label = Compose([AsDiscrete(to_onehot=n_classes)])

    # ---------------------------
    # Training Loop
    # ---------------------------
    for epoch in range(EPOCHS):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{EPOCHS}")
        model.train()
        epoch_loss = 0
        epoch_loss_seg = 0
        epoch_loss_feat = 0
        for step, batch_data in enumerate(train_loader, 1):
            if step > ITERATIONS:
                break
            
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            feature_targets = batch_data["features"].to(device)

            optimizer.zero_grad()

            seg_out, feat_out = model(inputs)

            loss_seg = dice_loss(seg_out, labels)
            loss_feat = feat_loss(feat_out, feature_targets)

            loss = loss_seg + lambda_feat * loss_feat

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_loss_seg += loss_seg.item()
            epoch_loss_feat += loss_feat.item()

            print(
                f"{step}/{ITERATIONS}, "
                f"seg: {loss_seg.item():.4f}, "
                f"feat: {loss_feat.item():.4f}"
            )

        epoch_loss /= ITERATIONS
        epoch_loss_seg /= ITERATIONS
        epoch_loss_feat /= ITERATIONS
        mlflow.log_metric("train_loss", epoch_loss, step=epoch + 1)
        mlflow.log_metric("train_loss_seg", epoch_loss_seg, step=epoch + 1)
        mlflow.log_metric("train_loss_feat", epoch_loss_feat, step=epoch + 1)

        # ---------------------------
        # Validation
        # ---------------------------
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)

                    roi_size = patch_size
                    sw_batch_size = 4

                    seg_out, _ = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model
                    )

                    seg_out = [post_pred(i) for i in decollate_batch(seg_out)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]

                    dice_metric(y_pred=seg_out, y=val_labels)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()

                mlflow.log_metric("val_mean_dice", metric, step=epoch + 1)

                if metric > best_metric:
                    best_metric = metric
                    torch.save(model.state_dict(), os.path.join(directory, "best_metric_model.pth"))
                    print("saved new best metric model")

                print(f"current mean dice: {metric:.4f}")
        
        # log learning rate
        mlflow.log_metric("learning_rate", optimizer.param_groups[0]["lr"], step=epoch + 1)

        # schedule step at the end of epoch
        scheduler.step()

    val_org_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            scaling,
            CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
        ]
    )

    val_org_ds = Dataset(data=val_files, transform=val_org_transforms)
    val_org_loader = DataLoader(val_org_ds, batch_size=1, num_workers=4)

    post_transforms = Compose(
        [
            Invertd(
                keys="pred",
                transform=val_org_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
                device="cpu",
            ),
            AsDiscreted(keys="pred", argmax=True, to_onehot=n_classes),
            AsDiscreted(keys="label", to_onehot=n_classes),
        ]
    )

    model.eval()

    with torch.no_grad():
        for val_data in val_org_loader:
            val_inputs = val_data["image"].to(device)
            roi_size = patch_size
            sw_batch_size = 4
            # val_data["pred"] = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
            seg_out, _ = sliding_window_inference(
                val_inputs, roi_size, sw_batch_size, model
            )

            val_data["pred"] = seg_out

            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)

            # Hausdorff (HD95)
            hausdorff_metric(y_pred=val_outputs, y=val_labels)

        # aggregate the final mean dice result
        metric_org = dice_metric.aggregate().item()
        # reset the status for next validation round
        dice_metric.reset()

        metric_hd95, _ = hausdorff_metric.aggregate()
        metric_hd95 = metric_hd95.item()
        hausdorff_metric.reset()

    print("Metric on original image spacing: ", metric_org)
    print("Mean HD95 (mm, original spacing): ", metric_hd95)

    mlflow.log_metric("val_mean_dice_original_spacing", metric_org, step=EPOCHS)
    mlflow.log_metric("val_mean_hd95_original_spacing", metric_hd95, step=EPOCHS)

if __name__ == "__main__":

    class_dict = {
        # "Task03_Liver": 3,
        "Task05_Prostate": 3,
        "Task06_Lung": 2,
        "Task09_Spleen": 2,
        "Task10_Colon": 2,    
    }

    patch_dict = {
        # "Task03_Liver": (128, 128, 96),
        "Task05_Prostate": (128, 128, 32),
        "Task06_Lung": (128, 128, 128),
        "Task09_Spleen": (128, 128, 128),
        "Task10_Colon": (128, 128, 128),    
    }

    channel_dict = {
        # "Task03_Liver": 1,
        "Task05_Prostate": 2,
        "Task06_Lung": 1,
        "Task09_Spleen": 1,
        "Task10_Colon": 1,    
    }

    for task in ["Task09_Spleen", "Task10_Colon", "Task05_Prostate", "Task06_Lung"]:

        print("-" * 20)
        print(f"Running task: {task}")
        print("-" * 20)

        # mlflow.set_tracking_uri("file:./mlruns")                    
        # mlflow.set_experiment(f"segmentation_{task}")

        mlflow.end_run()  # end previous run if any
        with mlflow.start_run(run_name=f"{task}_multitask_segmentation"):
            main(task=task, n_classes=class_dict[task], patch_size=patch_dict[task], n_channels=channel_dict[task])