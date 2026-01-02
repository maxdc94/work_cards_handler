import work_card_manager as wcm
import logging
from pathlib import Path
import costants as C

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


def main():

    logger.info("Reading pdf and trasform it in images on for each page")
    wcm.pdf_to_images(C.INPUT_PDF)

    _input = input("Do you want to enrich the dataset of images? (T/F)")

    if _input.lower() == 't':

        logger.info("Add raw data to training set work cards")
        folder = Path(C.WORK_CARDS_FOLDER)
        for img_path in folder.iterdir():
            if img_path.suffix.lower() in {".png"}:
                logger.info(f"Reading {img_path}")
                wcm.read_work_card(img_path)
                logger.info(f"Task completed for {img_path}")

    _input = input("Do you want to transfor the work cards into CSV file? (T/F)")

    if _input.lower() == 't':

        logger.info("Tranform cards to csv")
        folder = Path(C.WORK_CARDS_FOLDER)
        for img_path in folder.iterdir():
            if img_path.suffix.lower() in {".png"}:
                logger.info(f"Reading {img_path}")
                wcm.cards_to_CSV(img_path, C.MODEL)
                logger.info(f"Task completed for {img_path}")


if __name__ == "__main__":
    main()