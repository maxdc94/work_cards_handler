import cv2
import csv
import os
import re
import costants as C
import logging

logger = logging.getLogger(__name__)


TIME_REGEX = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")

# Carica label esistenti
labels = {}
if os.path.exists(C.TRAINING_SET_LABELS):
    with open(C.TRAINING_SET_LABELS, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["filename"]] = row["label"]

images = sorted(os.listdir(C.TRAINING_SET))

for img_name in images:
    if img_name in labels:
        continue  # già etichettata

    img_path = os.path.join(C.TRAINING_SET, img_name)
    img = cv2.imread(img_path)

    if img is None:
        logger.error(f"Error loading {img_name}")
        continue

    cv2.imshow("Labeling times (ESC to exit)", img)
    cv2.waitKey(1)

    while True:
        value = input(f"{img_name} → Enter time (hh:mm) or 's' to skip to next photo or 'q' to exit or 'd' to delte image: ").strip()

        value = value.replace(".", ":")

        if value.lower() == "s":
            cv2.destroyAllWindows()
            logger.info("Move to the next image...")
            break

        if value.lower() == "d":
            cv2.destroyAllWindows()
            logger.info("Deleting image...")
            os.remove(img_path)
            break

        if value.lower() == "q":
            cv2.destroyAllWindows()
            logger.info("Exit saving work...")
            break

        if TIME_REGEX.match(value):
            labels[img_name] = value
            break
        else:
            logger.warning("Invalid format. Use hh:mm")

    if value.lower() == "q":
        break

cv2.destroyAllWindows()

# CSV writing
with open(C.TRAINING_SET_LABELS, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])
    for filename, label in labels.items():
        writer.writerow([filename, label])

logger.info("Labeling completed")
