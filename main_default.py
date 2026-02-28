from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    NormalizeIntensityd,
    RandCropByLabelClassesd,
    EnsureTyped,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch, PersistentDataset
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import mlflow
from utils import set_deterministic, log_all_python_files
from itertools import cycle, islice
from monai.metrics import HausdorffDistanceMetric


EPOCHS = 1000
ITERATIONS = 100

def main(task: str, n_classes: int, patch_size: tuple, n_channels: int):
    
    torch.cuda.empty_cache()
    set_determinism(seed=0)
    set_deterministic()
    log_all_python_files()    

    mlflow.log_param("task", task)
    mlflow.log_param("n_classes", n_classes)
    mlflow.log_param("train_setup", "default_segmentation")

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
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            scaling,
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                num_classes=class_dict[task],
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            EnsureTyped(keys=["image", "label"]),
            # user can also add other random transforms
            # RandAffined(
            #     keys=['image', 'label'],
            #     mode=('bilinear', 'nearest'),
            #     prob=1.0, spatial_size=(128, 128, 128),
            #     rotate_range=(0, 0, np.pi/15),
            #     scale_range=(0.1, 0.1, 0.1)),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            scaling,
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                num_classes=class_dict[task],
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            EnsureTyped(keys=["image", "label"]),
            # user can also add other random transforms
            # RandAffined(
            #     keys=['image', 'label'],
            #     mode=('bilinear', 'nearest'),
            #     prob=1.0, spatial_size=(128, 128, 128),
            #     rotate_range=(0, 0, np.pi/15),
            #     scale_range=(0.1, 0.1, 0.1)),
        ]
    )

    # train_ds = CacheDataset(train_files, train_transforms, cache_rate=0.0, num_workers=4)
    train_ds = PersistentDataset(train_files, train_transforms, cache_dir="./data/cache_default")
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

    # val_ds = CacheDataset(val_files, val_transforms, cache_rate=0.0, num_workers=4)
    val_ds = PersistentDataset(val_files, val_transforms, cache_dir="./data/cache_default")
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    # standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda:0")
    model = UNet(
        spatial_dims=3,
        in_channels=n_channels,
        out_channels=n_classes,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        # norm=Norm.BATCH,
        norm=Norm.INSTANCE,
    ).to(device)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.SGD(model.parameters(), 1e-2, momentum=0.99, nesterov=True, weight_decay=3e-5)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=EPOCHS, power=0.9)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hausdorff_metric = HausdorffDistanceMetric(
        include_background=False,
        reduction="mean",
        percentile=95.0,   # HD95 (recommended)
        get_not_nans=True  # avoids NaN if class missing
    )

    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=n_classes)])
    post_label = Compose([AsDiscrete(to_onehot=n_classes)])

    for epoch in range(EPOCHS):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{EPOCHS}")
        model.train()

        epoch_loss = 0
        step = 0

        while step < ITERATIONS:

        # for step, batch_data in enumerate(train_loader, 1):
        #     if step > ITERATIONS:
        #         break
            step += 1
            batch_data = next(iter(train_loader))

            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            print(f"{step}/{ITERATIONS}, train_loss: {loss.item():.4f}")

        epoch_loss /= ITERATIONS
        epoch_loss_values.append(epoch_loss)

        mlflow.log_metric("train_loss", epoch_loss, step=epoch + 1)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = patch_size
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                mlflow.log_metric("val_mean_dice", metric, step=epoch + 1)
                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(directory, "best_metric_model.pth"))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
        
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
            roi_size = patch_dict[task]
            sw_batch_size = 4

            val_data["pred"] = sliding_window_inference(
                val_inputs, roi_size, sw_batch_size, model
            )

            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs, val_labels = from_engine(["pred", "label"])(val_data)

            # Dice
            dice_metric(y_pred=val_outputs, y=val_labels)

            # Hausdorff (HD95)
            hausdorff_metric(y_pred=val_outputs, y=val_labels)

        # ---- Aggregate Metrics ----
        metric_dice = dice_metric.aggregate().item()
        dice_metric.reset()

        metric_hd95, _ = hausdorff_metric.aggregate()
        metric_hd95 = metric_hd95.item()
        hausdorff_metric.reset()

    print("Mean Dice (original spacing): ", metric_dice)
    print("Mean HD95 (mm, original spacing): ", metric_hd95)

    mlflow.log_metric("val_mean_dice_original_spacing", metric_dice, step=EPOCHS)
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
        with mlflow.start_run(run_name=f"{task}_default_segmentation"):
            main(task=task, n_classes=class_dict[task], patch_size=patch_dict[task], n_channels=channel_dict[task])

