import pytesseract
def extract_text(image):
    # image is a PIL Image from app.py
    text = pytesseract.image_to_string(image)
    return text
