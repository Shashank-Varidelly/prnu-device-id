import json, torch, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, ".") 
sys.path.insert(0, "src")
from algorithms.cnn_classifier import PRNUResNet, train_one_epoch, evaluate, save_checkpoint
from utils.data_loader import PatchDataset, extract_patches
from torch.utils.data import DataLoader

SPLITS = "data/splits_demo.json"
RES_DIR = Path("data/processed/residuals/dresden")
CKPT_DIR = Path("checkpoints"); CKPT_DIR.mkdir(exist_ok=True)
CKPT_PATH = CKPT_DIR / "cnn_best.pth"
PATCH_SIZE = 128
PATCHES_PER_IMG = 4
BATCH = 32
EPOCHS = 6
LR = 1e-3

splits = json.loads(open(SPLITS).read())
devices = sorted(splits["dresden"]["train"].keys())
device_to_idx = {d: i for i, d in enumerate(devices)}
print("Devices:", devices)

def build_lists(split_dict, res_dir, device_to_idx):
    res_dir = Path(res_dir)
    paths, labels = [], []
    for device, img_paths in split_dict.items():
        lbl = device_to_idx[device]
        for img_path in img_paths:
            stem = Path(img_path).stem
            hits = list(res_dir.rglob(f"{stem}.npz"))
            if hits:
                paths.append(str(hits[0]))
                labels.append(lbl)
    return paths, labels

class NpzPatchDataset(PatchDataset):
    def __getitem__(self, idx):
        import torch
        img_idx, patch_num = self._index[idx]
        f = np.load(self.residual_paths[img_idx])
        residual = f["residual"].astype(np.float32) if "residual" in f else f[f.files[0]].astype(np.float32)
        if residual.ndim == 3:
          residual = residual.mean(axis=2)
        seed = self.rng.randint(0, 2**31) + patch_num
        patches = extract_patches(residual, patch_size=self.patch_size,
                                  max_patches=1, random_state=seed)
        patch = patches[0]
        tensor = (torch.from_numpy(patch).float().unsqueeze(0)
                  if patch.ndim == 2
                  else torch.from_numpy(patch).float().permute(2, 0, 1))
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, self.labels[img_idx]

train_paths, train_labels = build_lists(splits["dresden"]["train"], RES_DIR, device_to_idx)
val_paths, val_labels = build_lists(splits["dresden"]["val"], RES_DIR, device_to_idx)
print(f"Train: {len(train_paths)} residuals | Val: {len(val_paths)} residuals")
if len(train_paths) == 0:
    raise RuntimeError(f"No training residuals found under {RES_DIR}.")

train_ds = NpzPatchDataset(train_paths, train_labels, patch_size=PATCH_SIZE, patches_per_image=PATCHES_PER_IMG)
val_ds   = NpzPatchDataset(val_paths, val_labels, patch_size=PATCH_SIZE, patches_per_image=2)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", device)
model = PRNUResNet(num_classes=len(devices), in_channels=1, pretrained=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

best_val_acc = 0.0
start_epoch = 1
if CKPT_PATH.exists():
    ckpt        = torch.load(str(CKPT_PATH), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    best_val_acc = ckpt["metrics"].get("val_acc", 0.0)
    start_epoch  = ckpt.get("epoch", 0) + 1
    print(f"Resuming from epoch {start_epoch}, best_val_acc={best_val_acc:.3f}")
for epoch in range(1, EPOCHS + 1):
    tr = train_one_epoch(model, train_loader, optimizer, device)
    vl = evaluate(model, val_loader, device)
    scheduler.step()
    print(f"Epoch {epoch}/{EPOCHS} train_loss={tr['loss']:.4f} train_acc={tr['accuracy']:.3f} val_loss={vl['loss']:.4f} val_acc={vl['accuracy']:.3f}")
    if vl["accuracy"] >= best_val_acc:
        best_val_acc = vl["accuracy"]
        save_checkpoint(model, optimizer, epoch,
                        {"val_acc": best_val_acc, "num_classes": len(devices),
                         "device_names": devices, "patch_size": PATCH_SIZE},
                        str(CKPT_PATH))
        print(f"  => Checkpoint saved (val_acc={best_val_acc:.3f})")

print(f"\nDone. Best val_acc: {best_val_acc:.3f}")