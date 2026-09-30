import streamlit as st
import numpy as np
from PIL import Image

from ocr import extract_text
from embedding import create_embeddings
from attention import calculate_attention


st.title("🧠 Machine Learning")

st.write(
    "Upload an image to extract text "
    "and visualize attention scores."
)


file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if file:
    image = Image.open(file)

    st.image(image, width=500)

    text = extract_text(image)

    st.subheader("📝 Extracted Text")

    if not text.strip():
        st.error("No text found in the image.")
        st.stop()

    st.write(text)

    words = text.split()

    words = [
        word.strip(".,!?;:()[]{}")
        for word in words
    ]

    words = [
        word
        for word in words
        if len(word) > 2
    ]

    words = words[:20]

    if not words:
        st.error("No usable words found in the image.")
        st.stop()

    embeddings = create_embeddings(words)

    scores = calculate_attention(embeddings)

    st.subheader("🧠 Word Attention")

    display_scores = scores / scores.max()

    for word, score in zip(words, display_scores):
        st.write(f"**{word}**")
        st.progress(float(score))

    top_index = np.argmax(scores)

    top_word = words[top_index]

    st.success(
        f"⭐ Highest Attention: **{top_word}**"
    )
