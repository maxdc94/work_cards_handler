import cv2
import csv
import os
import re

IMAGE_DIR = "dataset/images"
CSV_PATH = "dataset/labels.csv"

TIME_REGEX = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")

# Carica label esistenti
labels = {}
if os.path.exists(CSV_PATH):
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["filename"]] = row["label"]

images = sorted(os.listdir(IMAGE_DIR))

for img_name in images:
    if img_name in labels:
        continue  # già etichettata

    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Errore caricando {img_name}")
        continue

    cv2.imshow("Labeling orari (ESC per uscire)", img)
    cv2.waitKey(1)

    while True:
        value = input(f"{img_name} → inserisci orario (hh:mm) oppure 's' per passare alla prossima foto o 'q' per uscire: ").strip()

        if value.lower() == "s":
            cv2.destroyAllWindows()
            print("Passa alla prossima immagine...")
            break

        if value.lower() == "q":
            cv2.destroyAllWindows()
            print("Uscita salvando il lavoro...")
            break

        if TIME_REGEX.match(value):
            labels[img_name] = value
            break
        else:
            print("❌ Formato non valido. Usa hh:mm (es. 07:56)")

    if value.lower() == "q":
        break

cv2.destroyAllWindows()

# Scrittura CSV con header
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])
    for filename, label in labels.items():
        writer.writerow([filename, label])

print("✅ Labeling completato")
