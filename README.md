# Fallen Tree – SPoTr Fine-tuning




## 1. Overview

This folder contains my implementation for the **fallen-tree part segmentation** task using the
**SPoTr (Self-positioning Point-based Transformer)** model.
This project uses the SPoTr model from the paper Self-Positioning Point-Based Transformer for Point Cloud Understanding (Park et al., CVPR 2023) and the official implementation at https://github.com/mlvlab/SPoTr.
I implemented a custom dataset loader for fallen vs. standing trees, a fine-tuning pipeline, evaluation, and prediction export scripts.

The pipeline:

1. Load the original **ShapeNetPart-pretrained SPoTr** checkpoint.
2. Fine-tune on the **fallen tree / standing tree** dataset (train split).
3. Evaluate on the **test split**.
4. Export per-sample NumPy prediction files whose filenames map back to the originals and
   whose last channel stores the predicted part label.

This README records **paths, commands, and output formats** so the experiment is fully reproducible.

---
## Project Structure

```bash
spotr-fallen-tree-part-segmentation/
├── README.md                     
├── fallen_tree_spotr_train.py
├── fallen_tree_spotr_eval.py
├── fallen_tree_spotr_export_preds.py
├── cfgs/
│   └── shapenetpart/
│       └── spotr.yaml            
├── requirements.txt              
└── .gitignore                    
```
---

## 2. Location & Environment

- Repo (SPoTr code):
  `/geoai_data/shared/saif/FallenTree_SPOTR/SPoTr`
- Data root:
  `/geoai_data/shared/saif/FallenTree_SPOTR/data`
- Python virtual env (inside repo):
  `.venv` 

Activate:

```bash
cd /geoai_data/shared/saif/FallenTree_SPOTR/SPoTr
source .venv/bin/activate
```

---


## 3. Dataset Layout

- Class-0 (standing trees):
/geoai_data/shared/saif/FallenTree_SPOTR/data/0/numpy/*.npy

- Class-1 (fallen trees):
/geoai_data/shared/saif/FallenTree_SPOTR/data/1/numpy/*.npy

Train / test split lists:
```bash
/geoai_data/shared/saif/FallenTree_SPOTR/data/train_test_split/shuffled_train_file_list.json
/geoai_data/shared/saif/FallenTree_SPOTR/data/train_test_split/shuffled_test_file_list.json
```


### Input numpy format

Each original point-cloud file has shape (N_points, 11) with columns:

- 0–2 -> x, y, z

- 3–5 -> R, G, B

- 6 -> intensity

- 7 -> raw class

- 8 -> instance

- 9 -> indices

- 10 -> part label (ground-truth segmentation)

The custom FallenTreeDataset in the scripts uses:

- Points per sample: NUM_POINTS = 2048 (random sampling or up-sampling with replacement).

- Features: [RGB (normalized to 0–1), intensity] -> 4 channels.

- Labels: part label from column 10.

- Class label: 0 = standing, 1 = fallen (based on folder).


---


## 4. Training – Fine-tuning SPoTr

- Script: fallen_tree_spotr_train.py

- Uses split "train" (approx. 849 samples).

- Config file: cfgs/shapenetpart/spotr.yaml

- Pretrained checkpoint (ShapeNetPart):
```bash
/geoai_data/shared/saif/FallenTree_SPOTR/checkpoints/ShapeNetPart/ckpt_best.pth
```
- Optimizer: AdamW (lr = 1e-4, weight_decay = 1e-4)

- Loss: CrossEntropyLoss over part labels.

- Epochs: 50

- Points per cloud: 2048

- Batch size: 2

- Checkpoints folder (fine-tuning run used for final results):
```bash
checkpoints_fallen_tree_run2/
```
- Best model selected manually and saved as:
```bash
checkpoints_fallen_tree_run2/spotr_fallen_tree_best.pth
```

### Run training
```bash
cd /geoai_data/shared/saif/FallenTree_SPOTR/SPoTr
source .venv/bin/activate

python fallen_tree_spotr_train.py
```

This will print per-epoch training loss and create:
```bash
checkpoints_fallen_tree_run2/spotr_fallen_tree_epoch{E}.pth
```


---

## 5. Evaluation

Script: fallen_tree_spotr_eval.py

- Split: "test" (213 samples).

- Loads checkpoint:
```bash
checkpoints_fallen_tree_run2/spotr_fallen_tree_best.pth
```

- Computes mean CE loss and per-point accuracy over a configurable number of batches.
```bash
cd /geoai_data/shared/saif/FallenTree_SPOTR/SPoTr
source .venv/bin/activate

python fallen_tree_spotr_eval.py

```
This script is mainly for sanity-checking training and the data pipeline.

---



## 6. Prediction Export

Script: fallen_tree_spotr_export_preds.py


What it does:

- Uses split "test" (all 213 samples).

- Builds SPoTr using cfgs/shapenetpart/spotr.yaml.

- Loads the fine-tuned checkpoint:
```bash
/geoai_data/shared/saif/FallenTree_SPOTR/SPoTr/checkpoints_fallen_tree_run2/spotr_fallen_tree_best.pth
```
- Runs inference over the full test set.

- Computes and prints:

  - Cross-entropy loss (per point, averaged over test set)

  - Per-point accuracy

- For each test sample, saves a NumPy file with:

  - File name tied to the original base name.

  - Coordinates + features + predicted label as final channel.

Run prediction export:
```bash
cd /geoai_data/shared/saif/FallenTree_SPOTR/SPoTr
source .venv/bin/activate

python fallen_tree_spotr_export_preds.py
```

Outputs:

- Directory with per-sample predictions:
```bash
/geoai_data/shared/saif/FallenTree_SPOTR/outputs/spotr_fallen_tree_test_preds/
```

---


## 7. Prediction File Naming & Format
### 7.1 Filename mapping

For each original test file:
```bash
<original_basename>.npy
```
the export script writes:
```bash
<original_basename>_pred.npy
```

Examples:

- fallen_trees_0.npy -> fallen_trees_0_pred.npy

- standing_trees_223.npy -> standing_trees_223_pred.npy

This gives a one-to-one mapping between original files and prediction files.

### 7.2 Array shape and channels

Each prediction file:

- is a NumPy array with shape (N_points, 8)

- channels:

  - 0–2: x, y, z

  - 3–6: normalized R, G, B, intensity
    (RGB values scaled to [0, 1])

- 7: predicted part segmentation label
   (integer-coded class index, stored as float32)



---


## 8. Final Test Metrics (best fine-tuned checkpoint)

Using fallen_tree_spotr_export_preds.py on the full test split (213 samples),
with checkpoint spotr_fallen_tree_best.pth, we obtain:

- Cross-entropy loss (per point, averaged over test set): 2.3191

- Per-point accuracy on test set: 54.68%

These values are also recorded in:
```bash
/geoai_data/shared/saif/FallenTree_SPOTR/SPoTr/fallen_tree_spotr_results.txt
```
which includes all key paths and the prediction array format for quick reference.


---
