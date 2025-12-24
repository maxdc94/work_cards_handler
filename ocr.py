import cv2
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
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
BATCH_SIZE = 16
EPOCHS = 500
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VOCAB = "0123456789:"
BLANK = 0
CHAR2IDX = {c: i + 1 for i, c in enumerate(VOCAB)}
IDX2CHAR = {i + 1: c for i, c in enumerate(VOCAB)}

# =========================
# PREPROCESSING
# =========================
def preprocess(img):
    if len(img.shape) == 2:  # già grayscale
        gray = img
    elif len(img.shape) == 3:  # BGR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Formato immagine non valido: {img.shape}")

    # ridimensiona all’input della rete
    gray = cv2.resize(gray, (IMG_W, IMG_H))
    gray = gray.astype("float32") / 255.0
    return gray

# =========================
# AUGMENTATION
# =========================
augment = A.Compose([
    A.RandomBrightnessContrast(0.2, 0.2),
    A.GaussNoise((0.05, 0.15)),
    A.MotionBlur(3),
    A.Affine(
        rotate=(-2, 2),
        translate_percent=(0.01, 0.03),
        scale=(0.95, 1.05)
    )
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
        img = cv2.imread(f"{self.img_dir}/{row.filename}")
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

        self.fc = nn.Linear(576, 128)
        self.rnn = nn.LSTM(128, 128, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(256, len(VOCAB) + 1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.mean(dim=3)
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
                text += IDX2CHAR[c]
            prev = c
        results.append(text)
    return results

def valid_time(t):
    if len(t) != 5 or t[2] != ":":
        return False
    h, m = t.split(":")
    return h.isdigit() and m.isdigit() and 0 <= int(h) <= 23 and 0 <= int(m) <= 59

def ctc_collate(batch):
    imgs = []
    labels = []
    label_lengths = []

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
def train():
    ds = TimeDataset("dataset/labels.csv", "dataset/images", True)
    dl = DataLoader(
        ds,
        BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        collate_fn=ctc_collate
    )

    model = CRNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0

        for imgs, labels, label_lengths in tqdm(dl):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            label_lengths = label_lengths.to(DEVICE)

            preds = model(imgs)
            preds = preds.permute(1, 0, 2)  # [T, B, C]

            pred_lengths = torch.full(
                (imgs.size(0),),
                preds.size(0),
                dtype=torch.long,
                device=DEVICE
            )

            preds = preds.log_softmax(2)

            loss = ctc(
                preds,
                labels,
                pred_lengths,
                label_lengths
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_sum += loss.item()

        print(f"EPOCH {epoch+1}: loss={loss_sum:.4f}")

    today = datetime.today().strftime("%Y%m%d")
    torch.save(model.state_dict(), f"ocr_timbratrice_{today}.pt")

# =========================
# INFERENCE
# =========================
def infer(image_path, model_path):
    model = CRNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # prendi 1 canale
    img = preprocess(img)  # restituisce [H, W]
    
    # replico 3 canali
    img = torch.tensor(img).unsqueeze(0).repeat(3, 1, 1)
    img = img.unsqueeze(0).to(DEVICE)  # [1, 3, H, W]

    with torch.no_grad():
        preds = model(img)
        preds = preds.log_softmax(2)

    text = decode(preds)[0]
    return text

def infer_folder(folder_path, model_path):
    # carica modello
    model = CRNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    results = {}
    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")) and "_c1" not in fname:
            path = os.path.join(folder_path, fname)
            try:
                time_text = infer(path, model)
                results[fname] = time_text
            except Exception as e:
                print(f"Errore con {fname}: {e}")
    return results

# =========================
if __name__ == "__main__":
    
    print("\n\n")
    _in = input("Do you want to train or to perform an inference? (T/I)")

    if _in == "T":
        train()
    elif _in == "I":
        # test single img
        text = infer("C:\\Users\\massimo\\Desktop\\leggi_cartellino\\tmp\\cells_imags\\pagina_002\\r16_c4.png", "ocr_timbratrice_V0.pt")
        print("ora stimata:" + text)
    else:
        print("Error")

    #infer_folder("C:\\Users\\massimo\\Desktop\\leggi_cartellino\\dataset\\images", "ocr_timbratrice_V0.pt")