import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import *
import time
from keras.models import load_model
from PIL import Image
from stillnessdetection import *
from fingers_trial import *
import threading


st.markdown("<h1 style='text-align: center; color: black;'>Elderly Monitoring System</h1>", unsafe_allow_html=True)


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
    st.write("Hand Tracking and Motion Detection in progress...")
    
    # Start each functionality in a separate thread
    thread1 = threading.Thread(target=alertFingers)
    thread2 = threading.Thread(target=stillnessdetection)
    
    # Start both threads
    thread1.start()
    thread2.start()
    
    # Wait for both threads to finish
    thread1.join()
    thread2.join()



    






    





