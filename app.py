import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import *
import time
from keras.models import load_model
from PIL import Image
from stillnessdetection import *
from alertdetection import *


st.markdown("<h1 style='text-align: center; color: black;'>Elderly Monitoring System</h1>", unsafe_allow_html=True)


col1, col2, col3, col4 = st.columns(4)

with col2:
    alert_sign_detection =  st.button("Alert Sign Detection")

with col3:
    stillness =  st.button("Stillness Detection")

if alert_sign_detection: 
    emergencySignalDetection()

if stillness:
    stillnessDetection()




















