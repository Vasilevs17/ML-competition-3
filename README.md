# ML Competition 3: Image Regression for Cover Scores

This repository contains a computer vision solution for an image-based regression competition. The task is to predict two continuous target values for each test image:

```text
c, s
```

The solution trains a neural network on labeled cover images and generates a submission file with predictions for the test set.

## Task

For every image from the test set, the model predicts two normalized scores:

```text
image_id, c, s
```

Both targets are treated as regression values in the `[0, 1]` range. The model applies a sigmoid output layer and clips final predictions to keep them inside the expected interval.

## Repository structure

```text
ML-competition-3/
├── README.md
├── requirements.txt
├── .gitignore
├── submission_ml_3 (1).csv
└── src/
    └── train_cover_regressor.py
```

The existing `submission_ml_3 (1).csv` file is a generated competition submission. The training script creates a cleaner output filename by default:

```text
submission_ml_3.csv
```

## Expected input files

The script expects the following files in the selected data directory:

```text
train_images_covers (1).zip
train_images_covers (2).zip
test_images_covers.zip
train_labels_covers.csv
sample_submission_covers.csv
```

The image archives are unpacked automatically if the extracted folders are not found.

## Approach

The solution is based on transfer learning for image regression:

- image archives are unpacked automatically;
- train labels are split into train and validation parts;
- images are loaded through a custom PyTorch `Dataset`;
- augmentations are applied during training;
- a pretrained ConvNeXt backbone from `timm` is used as the model base;
- the network predicts two continuous values at once;
- validation is monitored with Smooth L1 loss;
- early stopping keeps the best model state;
- test-time augmentation is used during inference.

## Model

The model uses:

```text
convnext_base.fb_in22k_ft_in1k
```

from the `timm` library. The final layer is configured for two regression outputs. A sigmoid activation is applied to produce values in the expected range.

## Training details

The main training settings are:

```text
image size: 384x384
batch size: 16
epochs: 18
optimizer: AdamW
learning rate: 1e-4
scheduler: OneCycleLR
loss: SmoothL1Loss
validation split: 10%
```

Data augmentation includes:

- random resized crop;
- horizontal flip;
- vertical flip;
- small rotation;
- color jitter;
- ImageNet normalization.

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training and prediction from the repository root:

```bash
python src/train_cover_regressor.py --data-dir . --output submission_ml_3.csv
```

For a smaller local test run, reduce the number of epochs:

```bash
python src/train_cover_regressor.py --data-dir . --output submission_ml_3.csv --epochs 2
```

## Main parameters

The script supports the following arguments:

```text
--data-dir       folder with archives and CSV files
--output         path to generated submission file
--seed           random seed
--batch-size     batch size
--epochs         maximum number of epochs
--image-size     input image resolution
--lr             learning rate
--weight-decay   AdamW weight decay
--patience       early stopping patience
--num-workers    number of DataLoader workers
--model-name     timm model name
```

Example:

```bash
python src/train_cover_regressor.py \
  --data-dir data/raw \
  --output submission_ml_3.csv \
  --epochs 10 \
  --batch-size 8
```

## Output

The final submission has the following columns:

```text
image_id, c, s
```

The script aligns predictions with `sample_submission_covers.csv`, fills missing predictions with `0.0` as a safety fallback, and saves the result to the selected output path.

## Main technologies

- Python
- PyTorch
- torchvision
- timm
- pandas
- NumPy
- Pillow

## Notes

This repository is intentionally focused on a single competition pipeline. The code keeps the full process in one readable script: data preparation, dataset creation, model training, validation, prediction and submission generation.