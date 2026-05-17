import json, torch, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, ".") 
sys.path.insert(0, "src")
from algorithms.siamese_network import SiameseNetwork, train_siamese_epoch, extract_embeddings
from algorithms.cnn_classifier import save_checkpoint
from utils.data_loader import SiamesePatchDataset, PatchDataset, extract_patches
from torch.utils.data import DataLoader

SPLITS  = "/content/splits_demo.json"
RES_DIR = Path("/content/residuals/dresden")
CKPT_DIR    = Path("checkpoints"); CKPT_DIR.mkdir(exist_ok=True)
CKPT_PATH   = CKPT_DIR / "siamese_best.pth"
GALLERY_EMB = Path("data/processed/gallery_embeddings.npy")
GALLERY_LBL = Path("data/processed/gallery_labels.json")
PATCH_SIZE  = 128
PAIRS_EPOCH = 8000
BATCH       = 32
EPOCHS      = 4
EMBED_DIM   = 128
MARGIN      = 1.0
DEVICES = ["Canon_Ixus55_0", "Canon_Ixus70_0", "Nikon_CoolPixS710_0", "Nikon_D200"]

splits = json.loads(open(SPLITS).read())
devices = DEVICES
device_to_idx = {d: i for i, d in enumerate(devices)}
print("Devices:", devices)

def build_lists(split_dict, res_dir, device_to_idx):
    res_dir = Path(res_dir)
    paths, labels = [], []
    for device, img_paths in split_dict.items():
        if device not in device_to_idx:
          continue
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
        img_idx, patch_num = self._index[idx]
        f = np.load(self.residual_paths[img_idx])
        residual = f["residual"].astype(np.float32) if "residual" in f else f[f.files[0]].astype(np.float32)
        if residual.ndim == 3:
            residual = residual.mean(axis=2)
        seed = self.rng.randint(0, 2**31) + patch_num
        patches = extract_patches(residual, patch_size=self.patch_size, max_patches=1, random_state=seed)
        patch = patches[0]
        tensor = torch.from_numpy(patch).float().unsqueeze(0)
        return tensor, self.labels[img_idx]

class NpzSiameseDataset(SiamesePatchDataset):
    def _load_random_patch(self, idx):
        f = np.load(self.residual_paths[idx])
        residual = f["residual"].astype(np.float32) if "residual" in f else f[f.files[0]].astype(np.float32)
        if residual.ndim == 3:
            residual = residual.mean(axis=2)
        patches = extract_patches(residual, patch_size=self.patch_size, max_patches=1,
                                  random_state=self.rng.randint(0, 2**31))
        return torch.from_numpy(patches[0]).float().unsqueeze(0)

train_paths, train_labels = build_lists(splits["dresden"]["train"], RES_DIR, device_to_idx)
print(f"Train: {len(train_paths)} residuals")
if len(train_paths) == 0:
    raise RuntimeError(f"No training residuals found under {RES_DIR}.")

train_ds   = NpzSiameseDataset(train_paths, train_labels, patch_size=PATCH_SIZE, pairs_per_epoch=PAIRS_EPOCH)
gallery_ds = NpzPatchDataset(train_paths, train_labels, patch_size=PATCH_SIZE, patches_per_image=4)
train_loader   = DataLoader(train_ds,   batch_size=BATCH, shuffle=True,  num_workers=4, pin_memory=True)
gallery_loader = DataLoader(gallery_ds, batch_size=256,   shuffle=False, num_workers=4, pin_memory=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", device)
model = SiameseNetwork(in_channels=1, embedding_dim=EMBED_DIM, pretrained=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_loss = float("inf")
start_epoch = 1
if CKPT_PATH.exists():
    ckpt        = torch.load(str(CKPT_PATH), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    best_loss   = ckpt["metrics"].get("loss", float("inf"))
    start_epoch = ckpt.get("epoch", 0) + 1
    print(f"Resuming from epoch {start_epoch}, best_loss={best_loss:.4f}")
for epoch in range(1, EPOCHS + 1):
    result = train_siamese_epoch(model, train_loader, optimizer, device, MARGIN)
    loss = result["loss"]
    print(f"Epoch {epoch}/{EPOCHS}  train_loss={loss:.4f}")
    if loss < best_loss:
        best_loss = loss
        save_checkpoint(model, optimizer, epoch,
                        {"loss": loss, "embedding_dim": EMBED_DIM,
                         "device_names": devices, "patch_size": PATCH_SIZE},
                        str(CKPT_PATH))
        print(f"  => Checkpoint saved (loss={loss:.4f})")

print("\nBuilding gallery embeddings...")
embeddings, labels = extract_embeddings(model, gallery_loader, device)
GALLERY_EMB.parent.mkdir(parents=True, exist_ok=True)
np.save(GALLERY_EMB, embeddings.numpy())
GALLERY_LBL.write_text(json.dumps({"labels": labels, "device_names": devices}))
print(f"Gallery: {embeddings.shape} embeddings, {len(labels)} labels")
print("Done.")