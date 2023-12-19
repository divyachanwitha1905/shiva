# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 11:38:43 2023

@author: DELL
"""

import streamlit as st
from PIL import Image
import model # This is your pre-trained model
import gdown

st.title('Steel Pipe Detection')

# URL of the model weights file on Google Drive
url = 'https://drive.google.com/uc?id=1J753l-T63J5oV-9rK6oJiO_F0RWXXZQk'


# Destination path where the downloaded file will be stored
output = 'model_weights.pth'

gdown.download(url, output, quiet=False)

# Now you can load the weights into your model
model.load_weights(output)

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    st.write("")
    st.write("Detecting...")
    
    # Here we would use our model to predict the output
    output_image, num_pipes, label = model.predict(image) # Assuming the model's predict function returns the output image, number of pipes and the label
    
    st.image(output_image, caption='Output Image with Detected Steel Pipes.', use_column_width=True)
    st.write(f"Number of steel pipes detected: {num_pipes}")
    st.write(f"Label of the image: {label}")
