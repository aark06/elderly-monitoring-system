import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import *
import time
from keras.models import load_model
from PIL import Image
from stillnessdetection import *


st.markdown("<h1 style='text-align: center; color: black;'>Elderly Monitoring System</h1>", unsafe_allow_html=True)
m = load_model('rsp4.h5')


f = st.file_uploader("Motion Detection", type=["jpg", "png"])
if f is not None:
    # st.write((type(f)))
    image = Image.open(f)
    image = np.asarray(image)
    image = cv2.resize(image, (224, 224))
    # image = image / 255.0
    image = image.reshape(1, 224, 224, 3)
    yhat1 = m.predict([image])
    mc1 = np.argmax(yhat1[0])
    st.image(image)
    # st.write(mc1)
    if(mc1 == 0):
        st.markdown("<p style='text-align: center; font-size: 20px;color: black;'>Prediction: Sitting</p>", unsafe_allow_html=True)
    if(mc1 == 1):
        st.markdown("<p style='text-align: center; font-size: 20px;color: black;'>Prediction: Sleeping</p>", unsafe_allow_html=True)
    if(mc1 == 2):
        st.markdown("<p style='text-align: center; font-size: 20px;color: black;'>Prediction: Alert Sign</p>", unsafe_allow_html=True)


col1, col2, col3 , col4, col5 = st.columns(5)

with col1:
    pass
with col2:
    pass
with col4:
    pass
with col5:
    pass
with col3 :
    stillness = st.button('Stillness Detection')

if stillness:
    stillnessdetection()


    






    





