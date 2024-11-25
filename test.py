import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model(r"D:\Monash Project\asl_model.h5")

# Mapping labels to alphabets
label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 
             8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 
             16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'}

st.title("Sign Language Detection")
st.text("Upload a hand gesture image or use the webcam for real-time predictions.")

# Webcam or Image Input
img_file_buffer = st.camera_input("Take a picture or use your webcam")

if img_file_buffer is not None:
    # Preprocess the input image
    img = Image.open(img_file_buffer)
    img = img.resize((28, 28))
    img = img.convert('L')
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    # Debugging
    st.write(f"Image Shape: {img.shape}")

    # Make prediction
    predictions = model.predict(img)
    predicted_label = np.argmax(predictions)
    predicted_letter = label_map[predicted_label]
    confidence = predictions[0][predicted_label] * 100

    # Display prediction
    st.write(f"Predicted Sign: {predicted_letter} (Confidence: {confidence:.2f}%)")


