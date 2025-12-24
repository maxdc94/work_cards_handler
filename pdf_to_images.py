import fitz              # PyMuPDF
import cv2
import numpy as np
import io
import os
from PIL import Image


def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_biggest_contour(img):
    """Trova il contorno più grande utile per ritagliare il contenuto."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    return max(contours, key=cv2.contourArea)



def extract_images_from_pdf(pdf_path, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_name = pdf_path.stem
    doc = fitz.open(pdf_path)

    for i, page in enumerate(doc, start=1):
        print(f"Processing page {i}...")

        # Render pagina come immagine
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Trova contorno principale
        cnt = detect_biggest_contour(img_cv)

        if cnt is None:
            print(f"⚠ Nessun contorno trovato nella pagina {i}, salto.")
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Ordina punti e ritaglia
        ordered = order_points(box.astype("float32"))
        w = int(rect[1][1])
        h = int(rect[1][0])
        dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype="float32")

        M = cv2.getPerspectiveTransform(ordered, dst)
        cropped = cv2.warpPerspective(img_cv, M, (w, h))

        cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

        out_path = os.path.join(output_dir, f"{pdf_name}_{i}.png")
        cv2.imwrite(out_path, cropped)

        print(f"✔ Salvata: {out_path}")

    doc.close()
    print("✔ Completato!")


if __name__ == "__main__":
    extract_images_from_pdf("input/cartellini_202512.pdf", "tmp/work_cards")
