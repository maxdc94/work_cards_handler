import cv2
import csv
import os
import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path
import ocr 
import shutil
import logging
import numpy as np
import costants as C

logger = logging.getLogger(__name__)

# Column coordinates (in pixels)
COLS = [
    (0, 105),    # first columns (day number)
    (150, 249),  # morning entrance
    (285, 394),  # morning exit
    (430, 538),  # afternoon entry
    (574, 682)   # afternoon exit
]

FIRST_ROW_Y = 679
ROW_HEIGHT = 54
NUM_ROWS = 31

def is_image_blank(image_path):
    std, darkest_mean = analyze_image(image_path)
    return  darkest_mean > 200

def analyze_image(image_path, border=15, top_n=10):
    """
    Crops the edges and calculates std + average of the top_n darkest pixels.

    :param image_path: image path
    :param border: number of pixels to crop at the edges
    :param top_n: number of darkest pixels to consider
    :return: std, average of the top_n darkest pixels
    """
    # Read grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    
    # Crop the edges
    img_cropped = img[border:-border, border:-border]
  
    # Calculate standard deviation
    std = np.std(img_cropped)

    # Average of the top_n darkest pixels
    pixels_sorted = np.sort(img_cropped.flatten())
    darkest_mean = np.mean(pixels_sorted[:top_n])

    #output_path = f"dbg/{darkest_mean}_{image_path.stem}.png"
    #cv2.imwrite(output_path, img_cropped)

    return std, darkest_mean

def read_work_card(img_path):

    file_name = img_path.stem
    
    os.makedirs(C.CELL_IMGS_FOLDER, exist_ok=True)

    path = os.path.join(C.CELL_IMGS_FOLDER, file_name)
    os.makedirs(path, exist_ok=True)

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rows = []

    for i in range(NUM_ROWS):
        y1 = FIRST_ROW_Y + i * ROW_HEIGHT
        y2 = y1 + ROW_HEIGHT


        for col_index, (x1, x2) in enumerate(COLS):
            cell = gray[y1:y2, x1:x2]


            cell_img = f"{file_name}_r{i:02d}_c{col_index}.png"
            cell_path = os.path.join(path, cell_img)

            if not os.path.exists(cell_path):
                cv2.imwrite(cell_path, cell)
            #else:
            #    logger.info(f"File already exists: {cell_path}")


            # add image to training set if its column is > 0 , and it is not empty and it is not already present in the dataset
            if col_index > 0:

                if not is_image_blank(cell_path):

                    training_path = os.path.join(C.TRAINING_SET, cell_img)
                    if not os.path.exists(training_path):
                        cv2.imwrite(training_path, cell)
                    else:
                        logger.info(f"File already exists: {training_path}")


    # save image in tmp folder
    shutil.copy(img_path, f"{path}/{file_name}.png")


def cards_to_CSV(img_path, model_path):

    file_name = img_path.stem

    os.makedirs(C.CELL_IMGS_FOLDER, exist_ok=True)

    path = os.path.join(C.CELL_IMGS_FOLDER, file_name)
    os.makedirs(path, exist_ok=True)

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rows = []

    for i in range(NUM_ROWS):
        y1 = FIRST_ROW_Y + i * ROW_HEIGHT
        y2 = y1 + ROW_HEIGHT

        row_values = []
        for col_index, (x1, x2) in enumerate(COLS):
            cell = gray[y1:y2, x1:x2]

            # cell image saving
            cell_img = f"{file_name}_r{i:02d}_c{col_index}.png"
            cell_path = os.path.join(path, cell_img)
            cv2.imwrite(cell_path, cell)

            if col_index == 0:
                row_values.append(i+1)
            else:
                text = ""
                if not is_image_blank(cell_path):
                    # OCR
                    text = ocr.infer(cell_path, model_path)
                    logger.debug("OCR {img_path} ==> [{i}, {col_index}] = {text}")
                row_values.append(text)
        rows.append(row_values)

    # save csv in tmp folder
    with open(f"{path}/{file_name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    # save image in tmp folder
    shutil.copy(img_path, f"{path}/{file_name}.png")

def pdf_to_images(pdf_path):



    file_name = Path(pdf_path).stem
    output_dir = f"{C.BASE_DIR}/tmp/work_cards"

    crop_rect = fitz.Rect(0, 0, 827, 300)

    doc = fitz.open(pdf_path)

    for i, page in enumerate(doc):
        pix = page.get_pixmap(clip=crop_rect, dpi=300)

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        img = img.rotate(-90, expand=True)

        img.save(f"{output_dir}/{file_name}_{i+1:03}.png")

    doc.close()

def test():
    folder = Path("tmp/cells_images/pagina_001")
    for img_path in folder.iterdir():
        if img_path.suffix.lower() in {".png"}:
            logger.info(f"Reading {img_path}")
            std, darkest_mean = analyze_image(img_path)
            
if __name__ == "__main__":
    test()