import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import csv

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

from tools import pad_to_min_size, random_crop, center_crop, remap_labels, dataset_structure

# Training dataset: Random crop every load to create different training value
class OpenEarthMapDataset(Dataset):
    def __init__(self, images, labels, crop_size=512, IGNORE=255):
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels
        self.crop_size = crop_size
        self.IGNORE = IGNORE

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        lab = Image.open(self.labels[idx])
        img_arr = np.array(img)
        lab_arr = np.array(lab)

        # Pad, random crop and re-label to make it consistence across batch
        cs = self.crop_size
        img_arr, lab_arr = pad_to_min_size(img_arr, lab_arr, cs, cs)
        img_arr, lab_arr = random_crop(img_arr, lab_arr, cs, cs)
        lab_arr = remap_labels(lab_arr, self.IGNORE)

        x = torch.from_numpy(img_arr).permute(2, 0, 1).float() / 255.0  # (3,cs,cs)
        y = torch.from_numpy(lab_arr).long()                             # (cs,cs)
        return x, y


# Validation dataset: Consistant (centered) crop to maintain validation dataset
class OpenEarthMapValDataset(OpenEarthMapDataset):
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        lab = Image.open(self.labels[idx])

        img_arr = np.array(img)
        lab_arr = np.array(lab)

        cs = self.crop_size
        img_arr, lab_arr = pad_to_min_size(img_arr, lab_arr, cs, cs)
        img_arr, lab_arr = center_crop(img_arr, lab_arr, cs, cs)
        lab_arr = remap_labels(lab_arr, self.IGNORE)

        x = torch.from_numpy(img_arr).permute(2,0,1).float()/255.0
        y = torch.from_numpy(lab_arr).long()
        return x, y

    
@torch.no_grad()
def pixel_accuracy_from_logits(logits, y, ignore_index=255):
    """
    Return the number of correct prediction and total prediction.
    logits (N,C,H,W): predicted values for y
    y (N,H,W): correct value
    """
    # logits: (N,C,H,W), y: (N,H,W)
    pred = logits.argmax(dim=1)
    mask = (y != ignore_index)
    correct = (pred[mask] == y[mask]).sum().item()
    total = mask.sum().item()
    return correct, total

def run_one_epoch_train(model, train_loader, optimizer, criterion, device):
    """
    Train one epoch.
    Return the average loss and accuracy.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        c, t = pixel_accuracy_from_logits(logits, y, ignore_index=255)
        correct += c
        total += t

    avg_loss = total_loss / len(train_loader.dataset)
    acc = correct / max(total, 1)
    return avg_loss, acc

@torch.no_grad()
def run_one_epoch_val(model, val_loader, criterion, device):
    """
    Run the model on the validation dataset.
    Return the average loss and accuracy.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(val_loader):
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)

        c, t = pixel_accuracy_from_logits(logits, y, ignore_index=255)
        correct += c
        total += t

    avg_loss = total_loss / len(val_loader.dataset)
    acc = correct / max(total, 1)
    return avg_loss, acc


def save_ckpt(path, model, optimizer, epoch, best_val_loss, history):
    """
    Save model Checkpoint.
    The checkpoint includes current epoch, states of model and optimizer (to resume training), best loss every, and history logs.
    """
    torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "history": history,  # optional but handy
            }, path)



def train_epochs(epochs, train_loader, val_loader, ckpt_dir = Path("checkpoints"), save_every_ckpt=False, IGNORE=255):
    """
    Train a specific number of epoch. Able to resume training from the last model.
    epoch (int): The number of epoch to train.
    train_loader (DataLoader): Train dataloader.
    val_loader (DataLoader): Validation dataloader.
    ckpt_dir (Path): Path to save the checkpoint.
    save_every_ckpt (bool): If False, save the model with best performance and last epoch only; if True, save models of every epochs.
    IGNORE (int): mask of invalid index.
    """

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"
    history_csv_path = ckpt_dir / "history.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE)

    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=8,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    if last_path.exists():
        # Resume from previous model
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        history = ckpt.get("history", [])
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from {last_path} at epoch {start_epoch} (best_val_loss={best_val_loss:.4f})")
    else:
        history = []
        best_val_loss = float("inf")
        start_epoch = 1
        
        
    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = run_one_epoch_train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = run_one_epoch_val(model, val_loader, criterion, device)
    
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        }
        history.append(row)
    
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%"
        )

    
        # Overwrite best checkpoint only if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_ckpt(best_path, model, optimizer, epoch, best_val_loss, history)
            print(f" New best model saved to {best_path} (best_val_loss={best_val_loss:.4f})")

        # Save last checkpoint
        save_ckpt(last_path, model, optimizer, epoch, best_val_loss, history)

        # Save every checkpoint if save_every_ckp
        if save_every_ckpt:
            cur_path = ckpt_dir / f"epoch{epoch}.pt"
            save_ckpt(cur_path, model, optimizer, epoch, best_val_loss, history)
        
        # Save history CSV each epoch 
        with open(history_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
            writer.writeheader()
            writer.writerows(history)
            
    print(f"Saved training history CSV: {history_csv_path}")
    return history




if __name__=="__main__":

    crop_size=512
    IGNORE = 255
    ckpt_dir = Path("checkpoints")

    # Get the directory list of images and labels 
    _, images, labels = dataset_structure()
    
    
    # Create train and validation data loader
    images_train, images_val, labels_train, labels_val = train_test_split(images, labels, test_size=0.1, random_state=42, shuffle=True)
    print("train:", len(images_train), "val:", len(images_val))

    train_ds = OpenEarthMapDataset(images_train, labels_train, crop_size=crop_size, IGNORE=IGNORE)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_ds   = OpenEarthMapValDataset(images_val, labels_val, crop_size=crop_size, IGNORE=IGNORE)
    val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)


   
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_every_ckpt', action="store_true")
    args = parser.parse_args()
    epochs = args.epochs
    save_every_ckpt = args.save_every_ckpt

    _ = train_epochs(epochs, train_loader, val_loader, ckpt_dir, save_every_ckpt, IGNORE)