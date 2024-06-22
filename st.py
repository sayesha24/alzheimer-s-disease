import streamlit as st
from keras.models import load_model
import numpy as np
import cv2
from PIL import Image

# Path to the trained model
model_path = '/Users/sayeshagoel/Desktop/inspirit ai /my_trained_model2.h5'

# Load the model
classifier = load_model(model_path)

# Define class labels
class_labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

def preprocess_image(image):
    # Convert the image to RGB and resize it
    image = image.convert('RGB')
    image = image.resize((128, 128))
    image_array = np.array(image, dtype=np.float32)
    image_array /= 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

def make_prediction(image_array):
    prediction = classifier.predict(image_array)
    result = np.argmax(prediction, axis=1)[0]
    return class_labels[result]

# Streamlit app
st.title("Alzheimer's Disease Detection")

st.header("Submit Brain MRI Scan")

uploaded_file = st.file_uploader("Upload an MRI scan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Scan', use_column_width=True)

    # Preprocess the image and make prediction
    image_array = preprocess_image(image)
    result_text = make_prediction(image_array)

    st.write(f"Prediction: {result_text}")

st.header("Submit Report on Biomarkers")
# You can add more functionalities related to biomarker reports here

# For example, add a text input for biomarker data
biomarker_data = st.text_area("Enter biomarker data here")

# Add a submit button for biomarker data
if st.button("Submit Biomarker Data"):
    st.write("Biomarker data submitted.")
