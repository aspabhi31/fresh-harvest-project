import streamlit as st
from model_helper import predict

st.title("Fresh Harvest AI Inspection")

uploaded_file = st.file_uploader("Upload the file", type=["jpg", "png"])
import tempfile
import os

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    ext = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.getbuffer())
        image_path = tmp.name
    prediction = predict(image_path)
    st.image(uploaded_file, caption="Uploaded File", width='stretch')
    st.info(f"Predicted Class: {prediction}")
else:
    st.write("Please upload an image file.")
