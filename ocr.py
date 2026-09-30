import os
import shutil
import pytesseract
from PIL import ImageOps

WINDOWS_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
WINDOWS_TESSDATA = r"C:\Program Files\Tesseract-OCR\tessdata"

# Use the Windows path only if Tesseract is not already on PATH
if shutil.which("tesseract") is None and os.path.exists(WINDOWS_TESSERACT):
    pytesseract.pytesseract.tesseract_cmd = WINDOWS_TESSERACT
    os.environ["TESSDATA_PREFIX"] = WINDOWS_TESSDATA


def preprocess(image):
    # convert to grayscale
    image = image.convert("L")

    # boost contrast so the ink stands out from the paper
    image = ImageOps.autocontrast(image)

    # upscale small images, Tesseract reads larger text better
    if image.width < 1500:
        scale = 1500 / image.width
        image = image.resize(
            (1500, int(image.height * scale))
        )

    return image


def extract_text(image):
    image = preprocess(image)
    return pytesseract.image_to_string(image, lang="eng")
