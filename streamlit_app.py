import streamlit as st
import face_recognition
import numpy as np
import cv2
from PIL import Image
from transformers import pipeline

st.title("Image Captioning + Face Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    face_locations = face_recognition.face_locations(image_np)
    image_with_boxes = image_np.copy()

    for top, right, bottom, left in face_locations:
        cv2.rectangle(image_with_boxes, (left, top), (right, bottom), (0, 255, 0), 2)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    with col2:
        st.image(image_with_boxes, caption="With Face Boxes", use_column_width=True)

    st.subheader("Caption")
    with st.spinner("Thinking..."):
        caption_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        caption = caption_pipeline(image)[0]["generated_text"]
        st.write(caption)