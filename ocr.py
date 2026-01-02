import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import albumentations as A
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split
import os
import cv2
import costants as C

# =========================
# CONFIG
# =========================
IMG_H = 40
IMG_W = 160
BATCH_SIZE = 16
EPOCHS = 2000
LR = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB = "0123456789:"
BLANK = 0
CHAR2IDX = {c: i + 1 for i, c in enumerate(VOCAB)}
IDX2CHAR = {i + 1: c for i, c in enumerate(VOCAB)}
EARLY_STOPPING_PATIENCE = 20

print("Using device:", DEVICE)

# =========================
# PREPROCESSING
# =========================
def preprocess(img):
    if len(img.shape) == 2:
        gray = img
    elif len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Formato immagine non valido: {img.shape}")
    gray = cv2.resize(gray, (IMG_W, IMG_H))
    gray = gray.astype("float32") / 255.0
    return gray

# =========================
# AUGMENTATION
# =========================
augment = A.Compose([
    A.RandomBrightnessContrast(0.1, 0.1),
    A.Affine(translate_percent=(0.02,0.02), scale=(0.98,1.02), rotate=(-2,2), mode=cv2.BORDER_CONSTANT, cval=255)
])

# =========================
# DATASET
# =========================
class OCRDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def encode(self, text):
        return torch.tensor([CHAR2IDX[c] for c in text], dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_text = self.samples[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = preprocess(img)
        if self.augment:
            img = augment(image=(img*255).astype(np.uint8))["image"]
            img = img.astype(np.float32)/255.0
        img = torch.tensor(img).unsqueeze(0).repeat(3,1,1)
        label = self.encode(label_text)
        return img, label

# =========================
# COLLATE FUNCTION
# =========================
def collate_fn(batch):
    imgs, labels, lengths = [], [], []
    for img, label in batch:
        imgs.append(img)
        labels.append(label)
        lengths.append(len(label))
    imgs = torch.stack(imgs)
    labels = torch.cat(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return imgs, labels, lengths

# =========================
# MODEL
# =========================
class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.cnn = backbone.features
        self.cnn[0][0].stride = (1,1)
        self.fc = nn.Linear(576,64)
        self.rnn = nn.LSTM(64,64,bidirectional=True,batch_first=True)
        self.classifier = nn.Linear(128,len(VOCAB)+1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0,3,1,2)        # [B,C,H,W] -> [B,W,H,C]
        x = x.mean(dim=3)             # global avg pool over H
        x = self.fc(x)
        x, _ = self.rnn(x)
        x = self.classifier(x)
        return x

# =========================
# DECODING
# =========================
def decode(pred):
    pred = pred.softmax(2).argmax(2)
    results = []
    for p in pred:
        text = ""
        prev = None
        for c in p:
            c = c.item()
            if c != prev and c != BLANK:
                text += IDX2CHAR.get(c,"")
            prev = c
        results.append(fix_time(text))
    return results

def fix_time(t:str)->str:
    t = "".join(c for c in t if c in "0123456789:")
    if t.count(":")!=1: return "00:00"
    h,m = t.split(":")
    if not(h.isdigit() and m.isdigit()): return "00:00"
    h = max(0,min(int(h),23))
    m = max(0,min(int(m),59))
    return f"{h:02d}:{m:02d}"

# =========================
# TRAIN FUNCTION
# =========================
def train(model, train_loader, val_loader):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5
    )
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        # ========= TRAIN =========
        model.train()
        train_loss_sum = 0.0

        for imgs, labels, lengths in tqdm(train_loader):
            imgs, labels, lengths = imgs.to(DEVICE), labels.to(DEVICE), lengths.to(DEVICE)

            preds = model(imgs)
            preds = preds.permute(1, 0, 2)  # [T,B,C]

            pred_lengths = torch.full(
                (imgs.size(0),), preds.size(0), dtype=torch.long, device=DEVICE
            )

            loss = ctc(preds.log_softmax(2), labels, pred_lengths, lengths)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss_sum += loss.item()

        train_loss = train_loss_sum / len(train_loader)

        # ========= VALIDATION =========
        model.eval()
        val_loss_sum = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels, lengths in val_loader:
                imgs, labels, lengths = imgs.to(DEVICE), labels.to(DEVICE), lengths.to(DEVICE)

                preds = model(imgs)
                preds_perm = preds.permute(1, 0, 2)

                pred_lengths = torch.full(
                    (imgs.size(0),), preds_perm.size(0), dtype=torch.long, device=DEVICE
                )

                loss = ctc(preds_perm.log_softmax(2), labels, pred_lengths, lengths)
                val_loss_sum += loss.item()

                decoded = decode(preds)
                for dec, lbl in zip(decoded, labels.split(lengths.tolist())):
                    label_text = "".join(IDX2CHAR[i.item()] for i in lbl)
                    if fix_time(dec) == fix_time(label_text):
                        correct += 1
                    total += 1

        val_loss = val_loss_sum / len(val_loader)
        val_acc = correct / total if total > 0 else 0.0

        print(
            f"Epoch {epoch+1}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f}"
        )

        scheduler.step(val_loss)

        # ========= EARLY STOPPING =========
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            today = datetime.today().strftime("%Y%m%d")
            torch.save(model.state_dict(), f"ocr_timbratrice_{today}.pt")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping!")
                break

# =========================
# INFERENCE
# =========================
def infer(image_path, model_path):
    model = CRNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path,map_location=DEVICE))
    model.eval()
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = preprocess(img)
    img = torch.tensor(img).unsqueeze(0).repeat(3,1,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        preds = model(img)
    return decode(preds)[0]

# =========================
# MAIN
# =========================
if __name__=="__main__":
    mode = input("Do you want to train or infer? (T/I): ").upper()
    if mode=="T":
        csv_path = C.TRAINING_SET_LABELS
        img_dir  = C.TRAINING_SET
        df = pd.read_csv(csv_path)
        samples = [(os.path.join(img_dir,row.filename), row.label) for _,row in df.iterrows()]
        train_samples, val_samples = train_test_split(samples,test_size=0.2,random_state=42)
        train_loader = DataLoader(OCRDataset(train_samples, augment=True), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader   = DataLoader(OCRDataset(val_samples, augment=False), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        model = CRNN().to(DEVICE)
        train(model, train_loader, val_loader)
    elif mode=="I":
        img_path = input("Path immagine: ")
        model_path = input("Path modello .pt: ")
        print("Predicted time:", infer(img_path, model_path))
    else:
        print("Input non valido")
