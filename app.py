import streamlit as st
from model_helper import predict

st.title("Fresh Harvest AI Inspection")

uploaded_file = st.file_uploader("Upload the file", type=["jpg", "png"])
import tempfile
import os

# Get the original extension (e.g., .png, .jpg)
ext = os.path.splitext(uploaded_file.name)[1]  # includes the dot
with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
    tmp.write(uploaded_file.getbuffer())
    image_path = tmp.name

prediction = predict(image_path)
