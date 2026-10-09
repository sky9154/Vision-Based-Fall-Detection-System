# Training tools

These tools implement the three-class workflow described in the [model training guide](../README.md). Run the commands from the `Train` directory. Replace `DATASET` with the dataset root; quote the path if it contains spaces.

The filename prefix gives the execution order. Files with the same prefix are alternatives or optional branches.

## Data contract

Every stage uses the same class labels:

| Label | Class |
| ---: | --- |
| `0` | Safe |
| `1` | Warning |
| `2` | Fall |

Pose CSV files contain the frame identifier, the integer label, and 26 coordinate features:

```text
frame,labels,x1,...,x13,y1,...,y13
```

A complete run produces this dataset structure:

```text
DATASET/
├── images/
│   ├── 0/
│   ├── 1/
│   └── 2/
├── augmented/                 # optional
│   ├── 0/
│   ├── 1/
│   └── 2/
├── preprocessed/              # optional
│   ├── equalized/{0,1,2}/
│   └── grayscale/{0,1,2}/
├── features/{0,1,2}.csv
├── interpolated/{0,1,2}.csv
├── sampled/{0,1,2}.csv
└── data.csv
```

## Run the pipeline

### 01 — Resize the Fall Detection Dataset

[`01_images_resize.py`](01_images_resize.py) reads each sequence's `rgb` directory, resizes its images to 640 × 480, and writes them to an `images` directory beside the source.

```bash
python tools/01_images_resize.py DATASET
```

This step is specific to the Fall Detection Dataset layout. It is not required for the UR Fall Detection Dataset.

### 02 — Assign the three class labels

Choose the classifier that matches the source dataset. Both commands collect the classified frames under `DATASET/images/{0,1,2}`.

For the Fall Detection Dataset:

```bash
python tools/02_images_classification.py DATASET
```

The source labels are mapped as follows:

| Source label | Project label |
| ---: | ---: |
| `1` | `0` — Safe |
| `2`, `4` | `1` — Warning |
| `3`, `5` | `2` — Fall |

For the UR Fall Detection Dataset:

```bash
python tools/02_images_classification_ur.py DATASET
```

The UR script reads `urfall-cam0-adls.csv` and `urfall-cam0-falls.csv` from the dataset root.

### 03 — Create an optional image variant

The pipeline can extract pose features from the classified images directly. Run a stage 03 tool only when the experiment calls for augmented or preprocessed input.

Geometric augmentation creates eight versions of each image under `DATASET/augmented/{0,1,2}`:

```bash
python tools/03_data_augmentation.py DATASET
```

Histogram equalization and grayscale conversion create separate datasets under `DATASET/preprocessed`:

```bash
python tools/03_image_equalization.py DATASET
```

Both tools accept `--source` and `--output` when the default directory names do not fit the experiment.

### 04 — Extract pose features

[`04_pose_get.py`](04_pose_get.py) letterboxes each image to 1280 × 720, runs MediaPipe Pose, and keeps 13 landmarks. Their X/Y coordinates become 26 features. A frame with no detected pose is recorded as 26 zeros for the interpolation stage.

Use one command that matches the chosen image source:

```bash
python tools/04_pose_get.py DATASET --source images
python tools/04_pose_get.py DATASET --source augmented
python tools/04_pose_get.py DATASET --source preprocessed/equalized
python tools/04_pose_get.py DATASET --source preprocessed/grayscale
```

The default output is `DATASET/features/{0,1,2}.csv`. Use `--output NAME` to keep feature sets from different experiments in separate directories.

The optional `--normalize` flag translates every skeleton so the hip midpoint is at the center of the feature canvas:

```bash
python tools/04_pose_get.py DATASET --source images --normalize
```

### 05 — Inspect a feature CSV

[`05_dataset_check.py`](05_dataset_check.py) prints descriptive statistics, then displays histograms and box plots for the 26 features.

```bash
python tools/05_dataset_check.py DATASET/features/0.csv
```

Run it again with `1.csv` and `2.csv` when comparing class distributions.

### 06 — Interpolate missing poses

[`06_linear_interpolation.py`](06_linear_interpolation.py) replaces zero coordinates with values interpolated from neighboring records. Leading and trailing gaps use the nearest available value. Any feature that contains no usable value remains zero.

```bash
python tools/06_linear_interpolation.py DATASET
```

The default input is `DATASET/features`, and the result is written to `DATASET/interpolated`.

### 07 — Balance the classes

[`07_data_sampled.py`](07_data_sampled.py) samples each class without replacement. With no sample count, it uses the size of the smallest class.

```bash
python tools/07_data_sampled.py DATASET
```

Set an explicit count or random seed when needed:

```bash
python tools/07_data_sampled.py DATASET --samples 10000 --random-state 1
```

The balanced files are written to `DATASET/sampled/{0,1,2}.csv`.

### 08 — Merge and shuffle the CSV files

[`08_merge_csv.py`](08_merge_csv.py) validates the required columns and class labels, merges the three files, and shuffles the rows.

```bash
python tools/08_merge_csv.py DATASET/sampled/0.csv DATASET/sampled/1.csv DATASET/sampled/2.csv --output DATASET/data.csv
```

The default shuffle seed is `42`; change it with `--random-state`.

### 09 — Train the LSTM

[`09_train_lstm.py`](09_train_lstm.py) performs a stratified train/test split and trains a 512-unit LSTM followed by a three-unit Softmax layer.

```bash
python tools/09_train_lstm.py DATASET/data.csv --output model/LSTM_03.h5
```

The defaults match the parameters in the parent training guide:

| Option | Default |
| --- | ---: |
| `--epochs` | 500 |
| `--batch-size` | 128 |
| `--test-size` | 0.2 |
| `--random-state` | 42 |
| `--hidden-units` | 512 |
| `--learning-rate` | 0.001 |

The model with the lowest validation loss is saved to the `--output` path. Training metrics are written beside it as `<model-name>_training.csv`. After training, the command prints the test loss, test accuracy, and the confusion matrix in **Safe**, **Warning**, **Fall** order.

[`09_lstm_model.ipynb`](09_lstm_model.ipynb) provides the notebook workflow. Its batch size is 1,280; use `--batch-size 1280` for a comparable CLI configuration.

### 10 — Run video inference

[`10_fall.py`](10_fall.py) loads a three-output model, extracts the same 13 landmarks from each video frame, and displays the class probabilities. Press `q` to stop playback.

```bash
python tools/10_fall.py --model model/LSTM_03.h5 --video video/fall.mp4
```

Use `--font` when the bundled Traditional Chinese font is not available at `font/NotoSansTC-Medium.otf`.
