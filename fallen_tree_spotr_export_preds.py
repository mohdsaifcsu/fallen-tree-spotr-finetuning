import os
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from openpoints.utils.config import EasyConfig
from openpoints.models import build_model_from_cfg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------
# 1. Dataset paths + constants
# ------------------------------------------------------------------
DATA_ROOT = "/geoai_data/shared/saif/FallenTree_SPOTR/data"
SPLIT_DIR = os.path.join(DATA_ROOT, "train_test_split")

NP_DIRS = {
    0: os.path.join(DATA_ROOT, "0", "numpy"),  # standing trees
    1: os.path.join(DATA_ROOT, "1", "numpy"),  # fallen trees
}

NUM_POINTS = 2048
BATCH_SIZE = 2

# Directory where we will save prediction numpy files
PRED_SAVE_DIR = "/geoai_data/shared/saif/FallenTree_SPOTR/outputs/spotr_fallen_tree_test_preds"
os.makedirs(PRED_SAVE_DIR, exist_ok=True)


# ------------------------------------------------------------------
# 2. Fallen-tree Dataset (with filenames)
# ------------------------------------------------------------------
class FallenTreeDataset(Dataset):
    """
    Fallen-tree point-cloud dataset for part segmentation.

    Each item returns:
      - coord     : [N, 3]  XYZ
      - feat      : [N, 4]  RGB + intensity (float32, 0–1)
      - seg_label : [N]     part labels (0..3 etc.)
      - cls_label : scalar  (0 = standing, 1 = fallen)
      - name      : original numpy filename (str)
    """

    def __init__(self, split="train", num_points=NUM_POINTS):
        super().__init__()
        self.num_points = num_points

        if split == "train":
            split_file = "shuffled_train_file_list.json"
        elif split == "test":
            split_file = "shuffled_test_file_list.json"
        else:
            raise ValueError(f"Unknown split: {split}")

        with open(os.path.join(SPLIT_DIR, split_file), "r") as f:
            file_list = json.load(f)

        self.samples = []
        for rel_path in file_list:
            sample_name = os.path.basename(rel_path)

            # Look in class-0 and class-1 numpy folders
            found = False
            for cls_id, np_dir in NP_DIRS.items():
                np_path = os.path.join(np_dir, sample_name)
                if os.path.exists(np_path):
                    # store npy path, class id, and original filename
                    self.samples.append((np_path, cls_id, sample_name))
                    found = True
                    break

            if not found:
                raise FileNotFoundError(
                    f"Could not find sample {sample_name} in 0/ or 1/ folders."
                )

        print(f"[Dataset] {split} split: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        np_path, cls_id, sample_name = self.samples[idx]
        arr = np.load(np_path)  # shape: (N_points, 11)

        # Columns: 0:X, 1:Y, 2:Z, 3:R, 4:G, 5:B, 6:Intensity, 7:RawClass,
        #          8:Instance, 9:Indices, 10:Part
        coords = arr[:, 0:3].astype(np.float32)
        rgb = arr[:, 3:6].astype(np.float32) / 255.0
        intensity = arr[:, 6:7].astype(np.float32)
        feat = np.concatenate([rgb, intensity], axis=1)          # [N, 4]
        seg_labels = arr[:, 10].astype(np.int64)                 # [N]

        N = coords.shape[0]
        if N >= self.num_points:
            choice = np.random.choice(N, self.num_points, replace=False)
        else:
            pad = np.random.choice(N, self.num_points - N, replace=True)
            choice = np.concatenate([np.arange(N), pad], axis=0)

        coords = coords[choice]
        feat = feat[choice]
        seg_labels = seg_labels[choice]

        return {
            "coord": torch.from_numpy(coords),          # (P, 3)
            "feat": torch.from_numpy(feat),             # (P, 4)
            "seg_label": torch.from_numpy(seg_labels),  # (P,)
            "cls_label": torch.tensor(cls_id, dtype=torch.long),
            "name": sample_name,
        }


def collate_fn(batch):
    """Stack per-sample dicts into a single batch and keep filenames."""
    coords = torch.stack([b["coord"] for b in batch], dim=0)       # (B, P, 3)
    feats = torch.stack([b["feat"] for b in batch], dim=0)         # (B, P, 4)
    seg_labels = torch.stack([b["seg_label"] for b in batch], 0)   # (B, P)
    cls_labels = torch.stack([b["cls_label"] for b in batch], 0)   # (B,)
    names = [b["name"] for b in batch]
    return coords, feats, seg_labels, cls_labels, names


# ------------------------------------------------------------------
# 3. Helper recursively wrap dicts in EasyConfig
# ------------------------------------------------------------------
def to_easy(obj):
    if isinstance(obj, EasyConfig):
        return obj
    if isinstance(obj, dict):
        return EasyConfig({k: to_easy(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_easy(v) for v in obj)
    return obj


# ------------------------------------------------------------------
# 4. Build SPoTr model from cfg + finetuned checkpoint
# ------------------------------------------------------------------
def build_spotr_model():
    cfg_path = os.path.join("cfgs", "shapenetpart", "spotr.yaml")
    ckpt_path = os.path.join(BASE_DIR, "checkpoints_fallen_tree_run2", "spotr_fallen_tree_best.pth")

    print("CFG path :", os.path.abspath(cfg_path))
    print("CKPT path:", os.path.abspath(ckpt_path))

    with open(cfg_path, "r") as f:
        raw_cfg = yaml.load(f, Loader=yaml.UnsafeLoader)

    # Convert nested dicts to EasyConfig for attribute-style access
    model_cfg = to_easy(raw_cfg["model"])
    cfg = EasyConfig(raw_cfg)
    cfg.model = model_cfg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nModel block NAME in cfg:", model_cfg.get("NAME", None))
    print("Encoder type            :", model_cfg.encoder_args.get("NAME", None))
    print("Using device:", device)

    model = build_model_from_cfg(model_cfg).to(device)
    model.eval()
    print("Model class:", model.__class__.__name__)
    if hasattr(model, "encoder"):
        print("Encoder class:", model.encoder.__class__.__name__)

    print("Loaded config file:", os.path.abspath(cfg_path))
    print("Loaded checkpoint :", os.path.abspath(ckpt_path))

    state = torch.load(ckpt_path, map_location=device)
    print("Checkpoint keys:", state.keys())

    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    print("Loaded checkpoint with strict=False")
    print("  Missing keys   :", missing)
    print("  Unexpected keys:", unexpected)

    print("\n[OK] SPoTr part-segmentation model is built, checkpoint loaded, and set to eval().")
    return model, device


# ------------------------------------------------------------------
# 5. Main run on test set and export predictions
# ------------------------------------------------------------------
def main():
    # Dataset + DataLoader for TEST split with no shuffle
    test_ds = FallenTreeDataset(split="test", num_points=NUM_POINTS)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Build model and load finetuned checkpoint
    model, device = build_spotr_model()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_points = 0
    num_batches = 0

    print(f"\n[INFO] Saving prediction files to: {PRED_SAVE_DIR}\n")

    model.eval()
    with torch.no_grad():
        for b_idx, (coords_b, feats_b, seg_b, cls_b, names_b) in enumerate(test_loader):
            # Move tensors to device
            coords_b = coords_b.to(device)      # (B, N, 3)
            feats_b  = feats_b.to(device)       # (B, N, 4)
            seg_b    = seg_b.to(device).long()  # (B, N)
            cls_b    = cls_b.to(device).long().view(-1, 1)  # (B, 1)

            # Build 7-channel feature [xyz | feats]
            x_b = torch.cat([coords_b, feats_b], dim=-1)  # (B, N, 7)

            batch = {
                "pos": coords_b,                              # (B, N, 3)
                "x":   x_b.permute(0, 2, 1).contiguous(),     # (B, 7, N)
                "y":   seg_b,                                 # (B, N)
                "cls": cls_b,                                 # (B, 1)
            }

            # Forward pass
            out = model(batch)
            logits = out["logits"] if isinstance(out, dict) and "logits" in out else out  # (B, C, N)

            loss = criterion(logits, seg_b)
            total_loss += loss.item()

            # Predictions
            preds_b = logits.argmax(dim=1)  # (B, N)
            total_correct += (preds_b == seg_b).sum().item()
            total_points += seg_b.numel()
            num_batches += 1

            # -----------------------------------------
            # Save per-sample prediction numpy files
            # -----------------------------------------
            coords_np = coords_b.cpu().numpy()   # (B, N, 3)
            feats_np  = feats_b.cpu().numpy()    # (B, N, 4)
            preds_np  = preds_b.cpu().numpy()    # (B, N)

            for i in range(coords_np.shape[0]):
                xyz_i = coords_np[i]                       # (N, 3)
                feat_i = feats_np[i]                       # (N, 4)
                pred_i = preds_np[i][..., None]            # (N, 1)

                # (N, 8): [x, y, z, r_norm, g_norm, b_norm, intensity, pred_label]
                data_i = np.concatenate(
                    [
                        xyz_i.astype(np.float32),
                        feat_i.astype(np.float32),
                        pred_i.astype(np.float32),
                    ],
                    axis=-1,
                )

                orig_name = names_b[i]  # for e.g, we can save this wayy "fallenTree_0123.npy"
                base_name = os.path.splitext(os.path.basename(orig_name))[0]
                out_name = f"{base_name}_pred.npy"
                out_path = os.path.join(PRED_SAVE_DIR, out_name)

                np.save(out_path, data_i)

            if (b_idx + 1) % 10 == 0:
                print(f"[Batch {b_idx+1}] Processed and saved predictions for batch.")

    avg_loss = total_loss / max(num_batches, 1)
    avg_acc = total_correct / max(total_points, 1)

    print("\n[DONE] Exported predictions for all test samples.")
    print(f"Eval CE loss (avg over {num_batches} batches): {avg_loss:.4f}")
    print(f"Per-point accuracy (avg): {avg_acc * 100:.2f}%")
    print("\n[NOTE] Each .npy file has shape (N_points, 8) with channels:")
    print("  0-2: x, y, z")
    print("  3-6: normalized R, G, B, intensity")
    print("  7  : predicted part label (float32, integer-coded)")


if __name__ == '__main__':
    main()
