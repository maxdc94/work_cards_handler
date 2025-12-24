import cv2
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
import albumentations as A
from tqdm import tqdm
from datetime import datetime
import os

# =========================
# CONFIG
# =========================
IMG_H = 40
IMG_W = 160
BATCH_SIZE = 8
EPOCHS = 500
LR = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

VOCAB = "0123456789:"
BLANK = 0
CHAR2IDX = {c: i + 1 for i, c in enumerate(VOCAB)}
IDX2CHAR = {i + 1: c for i, c in enumerate(VOCAB)}

EARLY_STOPPING_PATIENCE = 20

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
    A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.02, rotate_limit=2, border_mode=cv2.BORDER_CONSTANT, value=255)
])

# =========================
# DATASET
# =========================
class TimeDataset(Dataset):
    def __init__(self, csv_path, img_dir, training=True):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.training = training

    def __len__(self):
        return len(self.df)

    def encode(self, text):
        return torch.tensor([CHAR2IDX[c] for c in text], dtype=torch.long)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(os.path.join(self.img_dir, row.filename))
        img = preprocess(img)

        if self.training:
            img = augment(image=(img * 255).astype(np.uint8))["image"]
            img = img.astype(np.float32) / 255.0

        img = torch.tensor(img).unsqueeze(0).repeat(3, 1, 1)
        label = self.encode(row.label)
        return img, label

# =========================
# MODEL
# =========================
class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_small(pretrained=True)
        self.cnn = backbone.features
        self.cnn[0][0].stride = (1, 1)

        # FC + LSTM più piccolo
        self.fc = nn.Linear(576, 64)
        self.rnn = nn.LSTM(64, 64, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(128, len(VOCAB) + 1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.mean(dim=3)
        x = self.fc(x)
        x, _ = self.rnn(x)
        x = self.classifier(x)
        return x

# =========================
# DECODING ROBUSTO
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
                text += IDX2CHAR.get(c, "")
            prev = c
        # post-processing per renderlo HH:MM
        text = fix_time(text)
        results.append(text)
    return results

def fix_time(t):
    """
    Converte stringa generata dalla rete in HH:MM valida
    es: "12:1" -> "12:01", "::::" -> "00:00"
    """
    # Se non c'è due parti separate dai ':', ritorna 00:00
    if len(t) != 5 or t[2] != ":" or t == ":::::":
        return "00:00"
    h, m = t.split(":")
    if not (h.isdigit() and m.isdigit()):
        return "00:00"
    h = max(0, min(int(h), 23))
    m = max(0, min(int(m), 59))
    return f"{h:02d}:{m:02d}"

# =========================
# COLLATE
# =========================
def ctc_collate(batch):
    imgs, labels, label_lengths = [], [], []
    for img, label in batch:
        imgs.append(img)
        labels.append(label)
        label_lengths.append(len(label))

    imgs = torch.stack(imgs)
    labels = torch.cat(labels)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return imgs, labels, label_lengths

# =========================
# TRAIN
# =========================
def train(csv_path="dataset/labels.csv", img_dir="dataset/images"):
    ds = TimeDataset(csv_path, img_dir, True)
    val_size = max(1, int(len(ds) * 0.2))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=ctc_collate)
    val_dl = DataLoader(val_ds, BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=ctc_collate)

    model = CRNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0
        for imgs, labels, label_lengths in tqdm(train_dl):
            imgs, labels, label_lengths = imgs.to(DEVICE), labels.to(DEVICE), label_lengths.to(DEVICE)
            preds = model(imgs)
            preds = preds.permute(1,0,2)  # [T,B,C]
            pred_lengths = torch.full((imgs.size(0),), preds.size(0), dtype=torch.long, device=DEVICE)
            loss = ctc(preds.log_softmax(2), labels, pred_lengths, label_lengths)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss_sum += loss.item()

        # VALIDATION
        model.eval()
        val_loss_sum = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels, label_lengths in val_dl:
                imgs, labels, label_lengths = imgs.to(DEVICE), labels.to(DEVICE), label_lengths.to(DEVICE)
                preds = model(imgs)
                preds_perm = preds.permute(1,0,2)
                pred_lengths = torch.full((imgs.size(0),), preds_perm.size(0), dtype=torch.long, device=DEVICE)
                loss = ctc(preds_perm.log_softmax(2), labels, pred_lengths, label_lengths)
                val_loss_sum += loss.item()
                decoded = decode(preds)
                for dec, lbl in zip(decoded, labels.split(label_lengths.tolist())):
                    total += 1
                    label_text = "".join([IDX2CHAR[i.item()] for i in lbl])
                    if fix_time(dec) == fix_time(label_text):
                        correct += 1

        val_acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1}: train_loss={train_loss_sum:.4f}, val_loss={val_loss_sum:.4f}, val_acc={val_acc:.4f}")

        # Early stopping
        if val_loss_sum < best_val_loss:
            best_val_loss = val_loss_sum
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
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = preprocess(img)
    img = torch.tensor(img).unsqueeze(0).repeat(3,1,1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = model(img)
    return decode(preds)[0]

# =========================
if __name__ == "__main__":
    _in = input("Do you want to train or perform inference? (T/I): ").upper()
    if _in == "T":
        train()
    elif _in == "I":
        path = input("Path image: ")
        model_path = input("Path model .pt: ")
        text = infer(path, model_path)
        print("Ora stimata:", text)
    else:
        print("Input non valido")
