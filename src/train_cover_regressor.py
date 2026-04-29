"""Train an image regressor for the cover score prediction task.

The script prepares image folders from zip archives, trains a pretrained timm
backbone for two-output regression and creates a competition submission file.
"""

from __future__ import annotations

import argparse
import gc
import random
import zipfile
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class DataFiles(NamedTuple):
    train_zip_1: str = "train_images_covers (1).zip"
    train_zip_2: str = "train_images_covers (2).zip"
    test_zip: str = "test_images_covers.zip"
    labels_csv: str = "train_labels_covers.csv"
    sample_submission_csv: str = "sample_submission_covers.csv"


class TrainConfig(NamedTuple):
    seed: int
    batch_size: int
    epochs: int
    learning_rate: float
    image_size: int
    num_workers: int
    weight_decay: float
    patience: int
    model_name: str


TRAIN_DIR_NAME = "train_images_covers"
TEST_DIR_NAME = "test_images_covers"
TARGET_COLUMNS = ["c", "s"]


def set_seed(seed: int) -> None:
    """Make the training run more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unzip_if_needed(zip_path: Path, output_dir: Path, expected_subdir: str | None = None) -> None:
    """Extract a zip archive only when the expected output is not present."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Archive not found: {zip_path}")

    if expected_subdir is not None and (output_dir / expected_subdir).exists():
        return

    if expected_subdir is None and output_dir.exists() and any(output_dir.iterdir()):
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)


def prepare_data(data_dir: Path, data_files: DataFiles) -> tuple[Path, Path]:
    """Prepare train and test image folders from competition archives."""
    train_dir = data_dir / TRAIN_DIR_NAME
    test_dir = data_dir / TEST_DIR_NAME

    train_archives = [
        (data_files.train_zip_1, "train"),
        (data_files.train_zip_2, "train_2"),
    ]

    for archive_name, expected_subdir in train_archives:
        unzip_if_needed(data_dir / archive_name, train_dir, expected_subdir)

    unzip_if_needed(data_dir / data_files.test_zip, test_dir, "test")
    return train_dir, test_dir


def build_image_index(root_dir: Path) -> dict[str, Path]:
    """Map image ids to image paths."""
    image_index: dict[str, Path] = {}
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        for image_path in root_dir.rglob(pattern):
            image_index[image_path.stem] = image_path
    return image_index


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Create training and validation/test transforms."""
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_transform, eval_transform


class CoversDataset(Dataset):
    """Dataset for cover images and two regression targets."""

    def __init__(self, labels: pd.DataFrame, images_dir: Path, transform: transforms.Compose):
        self.labels = labels.reset_index(drop=True)
        self.transform = transform
        self.image_index = build_image_index(images_dir)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.labels.iloc[index]
        image_id = row["image_id"]
        image_path = self.image_index.get(image_id)

        if image_path is None:
            raise FileNotFoundError(
                f"Image for image_id={image_id} was not found. "
                "Check that image archives were extracted correctly."
            )

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        target = torch.tensor(row[TARGET_COLUMNS].values.astype("float32"))
        return image_tensor, target


class CoverRegressor(nn.Module):
    """Pretrained image backbone with two sigmoid regression outputs."""

    def __init__(self, model_name: str):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.backbone(images))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Evaluate the model using Smooth L1 loss."""
    model.eval()
    criterion = nn.SmoothL1Loss(beta=0.02)
    total_loss = 0.0
    total_items = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        predictions = model(images)
        loss = criterion(predictions, targets)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


def train_model(
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
) -> nn.Module:
    """Train the image regression model and keep the best validation state."""
    model = CoverRegressor(config.model_name).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.epochs,
        steps_per_epoch=max(len(train_loader), 1),
        pct_start=0.1,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    criterion = nn.SmoothL1Loss(beta=0.02)

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                predictions = model(images)
                loss = criterion(predictions, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / max(len(train_loader.dataset), 1)
        valid_loss = evaluate(model, valid_loader, device)
        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"train_loss={train_loss:.5f} | valid_loss={valid_loss:.5f}"
        )

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


@torch.no_grad()
def predict_with_tta(
    model: nn.Module,
    test_dir: Path,
    transform: transforms.Compose,
) -> pd.DataFrame:
    """Predict targets for test images with simple flip test-time augmentation."""
    model.eval()
    device = next(model.parameters()).device

    image_paths: list[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        image_paths.extend(test_dir.rglob(pattern))
    image_paths = sorted(image_paths)

    results: list[tuple[str, float, float]] = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        original_prediction = model(image_tensor).squeeze(0)
        horizontal_prediction = model(torch.flip(image_tensor, dims=[3])).squeeze(0)
        vertical_prediction = model(torch.flip(image_tensor, dims=[2])).squeeze(0)

        prediction = (
            (original_prediction + horizontal_prediction + vertical_prediction) / 3.0
        ).cpu().numpy()
        prediction = np.clip(prediction, 0.0, 1.0)

        results.append((image_path.stem, float(prediction[0]), float(prediction[1])))

    return pd.DataFrame(results, columns=["image_id", "c", "s"])


def build_submission(
    predictions: pd.DataFrame,
    sample_submission_path: Path,
    output_path: Path,
) -> None:
    """Align predictions with the sample submission and save the final CSV."""
    sample_submission = pd.read_csv(sample_submission_path)
    submission = sample_submission[["image_id"]].merge(predictions, on="image_id", how="left")
    submission[TARGET_COLUMNS] = submission[TARGET_COLUMNS].fillna(0.0)
    submission.to_csv(output_path, index=False)


def make_loaders(
    labels: pd.DataFrame,
    images_dir: Path,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders."""
    permutation = np.random.permutation(len(labels))
    split_index = int(len(labels) * 0.9)
    train_indices = permutation[:split_index]
    valid_indices = permutation[split_index:]

    train_labels = labels.iloc[train_indices].reset_index(drop=True)
    valid_labels = labels.iloc[valid_indices].reset_index(drop=True)

    train_dataset = CoversDataset(train_labels, images_dir, train_transform)
    valid_dataset = CoversDataset(valid_labels, images_dir, eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, valid_loader


def run_pipeline(data_dir: Path, output_path: Path, config: TrainConfig) -> None:
    """Run the full training and prediction pipeline."""
    set_seed(config.seed)
    data_files = DataFiles()

    print("Preparing data...")
    train_images_dir, test_images_dir = prepare_data(data_dir, data_files)
    labels = pd.read_csv(data_dir / data_files.labels_csv)
    train_transform, eval_transform = build_transforms(config.image_size)

    train_loader, valid_loader = make_loaders(
        labels=labels,
        images_dir=train_images_dir,
        train_transform=train_transform,
        eval_transform=eval_transform,
        config=config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = train_model(train_loader, valid_loader, device, config)
    predictions = predict_with_tta(model, test_images_dir, eval_transform)
    build_submission(
        predictions=predictions,
        sample_submission_path=data_dir / data_files.sample_submission_csv,
        output_path=output_path,
    )

    print(f"Saved submission to: {output_path}")

    del model, train_loader, valid_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ConvNeXt-based image regressor and create a submission file.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("submission_ml_3.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--model-name",
        type=str,
        default="convnext_base.fb_in22k_ft_in1k",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        image_size=args.image_size,
        num_workers=args.num_workers,
        weight_decay=args.weight_decay,
        patience=args.patience,
        model_name=args.model_name,
    )
    run_pipeline(data_dir=args.data_dir, output_path=args.output, config=config)


if __name__ == "__main__":
    main()
