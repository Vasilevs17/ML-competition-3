#!/usr/bin/env python
# coding: utf-8
"""
Скрипт для обучения модели и формирования submission.csv.
Запускать в Google Colab после загрузки архивов и CSV в рабочую директорию.
"""

import gc
import os
import random
import zipfile
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
except ImportError as exc:  # pragma: no cover - для Colab
    raise SystemExit(
        "Не найдены зависимости PyTorch/torchvision. Установите их в Colab: "
        "!pip install torch torchvision"
    ) from exc

try:
    import timm
except ImportError:
    print("Устанавливаем timm...")
    os.system("pip -q install timm")
    import timm


SEED = 42
BATCH_SIZE = 16
NUM_EPOCHS = 18
LR = 1e-4
IMG_SIZE = 384
NUM_WORKERS = 2
WEIGHT_DECAY = 1e-4
PATIENCE = 6

TRAIN_ZIPS = [
    ("train_images_covers (1).zip", "train"),
    ("train_images_covers (2).zip", "train_2"),
]
TRAIN_DIR = Path("train_images_covers")
TEST_ZIP = "test_images_covers.zip"
TEST_DIR = Path("test_images_covers")
LABELS_CSV = "train_labels_covers.csv"
SUBMISSION_SAMPLE = "sample_submission_covers.csv"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unzip_if_needed(zip_path: str, out_dir: Path, expected_subdir: str | None = None) -> None:
    if not Path(zip_path).exists():
        raise FileNotFoundError(f"Не найден архив: {zip_path}")
    if expected_subdir:
        if (out_dir / expected_subdir).exists():
            return
    else:
        if out_dir.exists() and any(out_dir.iterdir()):
            return
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def prepare_data() -> Tuple[Path, Path]:
    for zip_name, expected_subdir in TRAIN_ZIPS:
        unzip_if_needed(zip_name, TRAIN_DIR, expected_subdir)
    if not TEST_DIR.exists() or not any(TEST_DIR.iterdir()):
        unzip_if_needed(TEST_ZIP, TEST_DIR, "test")
    return TRAIN_DIR, TEST_DIR


def build_image_index(root_dir: Path) -> dict:
    image_index = {}
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        for img_path in root_dir.rglob(ext):
            image_index[img_path.stem] = img_path
    return image_index


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tfms, val_tfms


class CoversDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: Path, transform: transforms.Compose):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.image_index = build_image_index(images_dir)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        img_path = self.image_index.get(image_id)
        if img_path is None:
            raise FileNotFoundError(
                f"Не найден файл изображения для id={image_id}. "
                "Проверьте, что архивы распакованы корректно."
            )
        from PIL import Image

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        target = torch.tensor([row["c"], row["s"]], dtype=torch.float32)
        return image, target


class CoverRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_base.fb_in22k_ft_in1k", pretrained=True, num_classes=2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return torch.sigmoid(x)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    criterion = nn.SmoothL1Loss(beta=0.02)
    total_loss = 0.0
    total = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        preds = model(images)
        loss = criterion(preds, targets)
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
    return total_loss / max(total, 1)


def train_model(train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> nn.Module:
    model = CoverRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=NUM_EPOCHS,
        steps_per_epoch=max(len(train_loader), 1),
        pct_start=0.1,
    )
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    else:
        scaler = torch.amp.GradScaler("cpu", enabled=False)
    criterion = nn.SmoothL1Loss(beta=0.02)

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                preds = model(images)
                loss = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)
            scheduler.step()

        train_loss = running_loss / max(len(train_loader.dataset), 1)
        val_loss = evaluate(model, val_loader, device)
        print(
            f"Эпоха {epoch + 1}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Ранняя остановка по patience.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model: nn.Module, test_dir: Path, transform: transforms.Compose) -> pd.DataFrame:
    model.eval()
    device = next(model.parameters()).device
    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        image_paths.extend(test_dir.rglob(ext))
    image_paths = sorted(image_paths)
    results: List[Tuple[str, float, float]] = []

    from PIL import Image

    for img_path in image_paths:
        image_id = img_path.stem
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        preds = model(image_tensor).squeeze(0)
        hflip = torch.flip(image_tensor, dims=[3])
        vflip = torch.flip(image_tensor, dims=[2])
        preds_hflip = model(hflip).squeeze(0)
        preds_vflip = model(vflip).squeeze(0)
        preds = ((preds + preds_hflip + preds_vflip) / 3.0).cpu().numpy()
        preds = np.clip(preds, 0.0, 1.0)
        results.append((image_id, float(preds[0]), float(preds[1])))

    return pd.DataFrame(results, columns=["image_id", "c", "s"])


def main() -> None:
    set_seed(SEED)
    train_images_dir, test_images_dir = prepare_data()

    labels = pd.read_csv(LABELS_CSV)
    train_tfms, val_tfms = build_transforms()

    # Трен/вал сплит
    perm = np.random.permutation(len(labels))
    split = int(len(labels) * 0.9)
    train_idx, val_idx = perm[:split], perm[split:]
    train_df = labels.iloc[train_idx].reset_index(drop=True)
    val_df = labels.iloc[val_idx].reset_index(drop=True)

    train_dataset = CoversDataset(train_df, train_images_dir, train_tfms)
    val_dataset = CoversDataset(val_df, train_images_dir, val_tfms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    model = train_model(train_loader, val_loader, device)

    preds_df = predict(model, test_images_dir, val_tfms)
    sample = pd.read_csv(SUBMISSION_SAMPLE)
    submission = sample[["image_id"]].merge(preds_df, on="image_id", how="left")
    submission[["c", "s"]] = submission[["c", "s"]].fillna(0.0)
    submission.to_csv("submission_ml_3.csv", index=False)
    print("Готово! submission_ml_3.csv сохранен.")
    del model, train_loader, val_loader, train_dataset, val_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
