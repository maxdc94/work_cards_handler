# Work Cards Handler 🛠️

[![License: GPL](https://img.shields.io/badge/License-GPL-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)

**Work Cards Handler** is a specialized tool designed to automate the extraction of data from physical work cards (punched cards or attendance sheets) scanned into PDF format. It uses Computer Vision and a Deep Learning OCR model (CRNN) to recognize and digitize timestamps.

## 🌟 Key Features

* **PDF Processing:** Automatically converts PDF scans into processed images, detecting and cropping the relevant work card area.
* **Cell Extraction:** Segments the work card grid into individual cells (e.g., entry/exit times) for targeted analysis.
* **Custom OCR (CRNN):** Uses a Convolutional Recurrent Neural Network built with PyTorch (MobileNetV3 + LSTM) specifically for reading "HH:MM" timestamps.
* **Dataset Tooling:** Includes a CLI utility (`tag_images.py`) for manual labeling and enriching the training dataset.
* **CSV Export:** Transforms visual data from work cards into structured CSV files.

---

## 🏗️ Project Structure

* `main.py`: The entry point that orchestrates the workflow from PDF to CSV.
* `work_card_manager.py`: Handles image manipulation, cell segmentation, and OCR coordination.
* `ocr.py`: The Deep Learning engine containing the CRNN model architecture and training logic.
* `pdf_to_images.py`: Manages the initial conversion and perspective correction of PDF pages.
* `tag_images.py`: A utility for manual image labeling using OpenCV.
* `costants.py`: Centralized configuration for paths and model settings.

---

## 🛠️ Tech Stack

* **Linguaggio:** Python 3.x
* **Deep Learning:** PyTorch, Torchvision
* **Computer Vision:** OpenCV (cv2), Pillow, Albumentations
* **PDF Handling:** PyMuPDF (fitz)

---

## 🚀 Getting Started

### Prerequisites
You will need a Python environment with the following dependencies:
```bash
pip install torch torchvision opencv-python pymupdf albumentations pandas tqdm
```

## Installation
Clone the repository:

Bash

git clone [https://github.com/maxdc94/work_cards_handler.git](https://github.com/maxdc94/work_cards_handler.git)
cd work_cards_handler
Configuration: Verify the paths in costants.py to ensure INPUT_PDF and MODEL point to the correct files.

Usage: Run the main script:

```bash
python main.py
```

* The script will interactively guide you through:
* Converting the PDF into images.
* Enriching the training dataset by extracting single cells.
* Transforming the work cards into structured CSV files using the OCR model.

## 🧠 OCR Model Training
To train or refine the model on your specific card format:

**Labeling:** Run tag_images.py to manually label the images stored in the training set folder.

**Training:** Run ocr.py and select Train (T).

The model uses a CRNN (MobileNetV3 + LSTM) architecture.

It includes Early Stopping to save the best performing model automatically.

**Inference:** Use the Infer (I) mode in ocr.py to test the model on a single image.

## 📄 License
This project is licensed under the GPL v2.0 License.

## ✉️ Contacts
Massimo: https://github.com/maxdc94