from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL = f"{BASE_DIR}/models/ocr_timbratrice_20260102_078.pt"
INPUT_PDF = f"{BASE_DIR}/input/cartellini_202511.pdf"
WORK_CARDS_FOLDER = f"{BASE_DIR}/tmp/work_cards"
CELL_IMGS_FOLDER = f"{BASE_DIR}/tmp/cells_images"
TRAINING_SET = f"{BASE_DIR}/dataset/images"
TRAINING_SET_LABELS = f"{BASE_DIR}/dataset/labels.csv"