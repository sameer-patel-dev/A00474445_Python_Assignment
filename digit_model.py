import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from utils import load_or_train_model, predict_digit
from PIL import Image, ImageOps


def show_page():
    st.header("Digit Classifier")
    st.write("This app predicts digits from 0 to 9")

    model = load_or_train_model()
    uploaded_file = st.file_uploader("Upload an image of a digit", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        label, confidence = predict_digit(image, model)
        st.success(f'Prediction: {label} with confidence {confidence:.2f}')
        st.image(image, caption='Uploaded Digit', use_column_width=True)
        

    else:
        st.write("Please upload an image file to predict the digit.")


    

if __name__ == "__main__":
    show_page()
