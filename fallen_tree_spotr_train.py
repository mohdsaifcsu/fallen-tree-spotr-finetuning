import os
import json
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from openpoints.utils.config import EasyConfig
from openpoints.models import build_model_from_cfg


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


# ------------------------------------------------------------------
# 2. Fallen-tree Dataset
# ------------------------------------------------------------------
class FallenTreeDataset(Dataset):
    """
    Fallen-tree point-cloud dataset for part segmentation.

    Each item returns:
      - coord     : [N, 3]  XYZ
      - feat      : [N, 4]  RGB + intensity (float32, 0–1)
      - seg_label : [N]     part labels (0..3 etc.)
      - cls_label : scalar  (0 = standing, 1 = fallen)
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
                    self.samples.append((np_path, cls_id))
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
        np_path, cls_id = self.samples[idx]
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
        }


def collate_fn(batch):
    """Stack per-sample dicts into a single batch."""
    coords = torch.stack([b["coord"] for b in batch], dim=0)       # (B, P, 3)
    feats = torch.stack([b["feat"] for b in batch], dim=0)         # (B, P, 4)
    seg_labels = torch.stack([b["seg_label"] for b in batch], 0)   # (B, P)
    cls_labels = torch.stack([b["cls_label"] for b in batch], 0)   # (B,)
    return coords, feats, seg_labels, cls_labels


# ------------------------------------------------------------------
# 3. Helper: recursively wrap dicts in EasyConfig
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
# 4. Build SPoTr model from cfg + checkpoint
# ------------------------------------------------------------------
def build_spotr_model():
    cfg_path = os.path.join("cfgs", "shapenetpart", "spotr.yaml")
    ckpt_path = os.path.join("..", "checkpoints", "ShapeNetPart", "ckpt_best.pth")
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

    state = torch.load(ckpt_path, map_location=device)
    print("Checkpoint keys:", state.keys())

    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    print("Loaded checkpoint with strict=False")
    print("  Missing keys :", missing)
    print("  Unexpected keys:", unexpected)

    print("\n[OK] SPoTr part-segmentation model is built, checkpoint loaded, and set to eval().")
    return model, device


# ------------------------------------------------------------
# 5. Main: training loop for fine-tuning on fallen-tree data
# ------------------------------------------------------------
def main():
    # Dataset + loader
    train_ds = FallenTreeDataset(split="train", num_points=NUM_POINTS)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Build model
    model, device = build_spotr_model()
    model.train()  # put in training mode
    print("Model is now in TRAIN mode")

    # Optimizer + loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    ckpt_dir = "checkpoints_fallen_tree_run2"
    os.makedirs(ckpt_dir, exist_ok=True)


    NUM_EPOCHS = 50  # we can increase if we want retrain more

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        num_batches = 0

        for coords_b, feats_b, seg_b, cls_b in train_loader:

            # ---- move to device ----
            coords_b = coords_b.to(device)              # (B, N, 3)
            feats_b  = feats_b.to(device)               # (B, N, 4)
            seg_b    = seg_b.to(device).long()          # (B, N)
            cls_b    = cls_b.to(device).long().view(-1, 1)  # (B, 1)

            # ---- build 7-channel feature ----
            x_b = torch.cat([coords_b, feats_b], dim=-1)    # (B, N, 7)

            # ---- batch identical to sanity script ----
            batch = {
               "pos": coords_b,                              # (B, N, 3)
               "x"  : x_b.permute(0, 2, 1).contiguous(),     # (B, 7, N)
               "y"  : seg_b,                                 # (B, N)
               "cls": cls_b,                                 # (B, 1)
            }

            # debug shapes only for first batch
            # if epoch == 0:
            #     print("coords_b:", coords_b.shape)
            #     print("x_b:", x_b.shape)
            #     print("seg_b:", seg_b.shape)
            #     print("cls_b:", cls_b.shape)
            #     break
            # -----------------------------
            # 4) Forward + loss
            # -----------------------------
            out = model(batch)
            logits = out["logits"] if isinstance(out, dict) else out  # (B, C, N)

            loss = criterion(logits, seg_b)  # Cross-entropy over parts

            # -----------------------------
            # 5) Backprop + update
            # -----------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        epoch_loss = running_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - train loss: {epoch_loss:.4f}")
        save_path = os.path.join(ckpt_dir, f"spotr_fallen_tree_epoch{epoch+1}.pth")
        torch.save(
            {
                 "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "epoch": epoch + 1,
                 "loss": epoch_loss,
            },
            save_path,
        )
        print(f"Saved checkpoint: {save_path}")



if __name__ == "__main__":
    main()

