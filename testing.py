# testing the model on single images
from tensorflow.keras.models import load_model
import numpy as np
import cv2

image1 = cv2.imread('images/Sitting/7_700.jpg')
image2 = cv2.imread('images/Sleeping/2_36.jpg')
image3 = cv2.imread('images/Waving/3_84.jpg')
m = load_model('rsp4.h5')

image1 = np.asarray(image1)
image1 = cv2.resize(image1, (224, 224))
image1 = image1.reshape(1,224,224,3)

image2 = np.asarray(image2)
image2 = cv2.resize(image2, (224, 224))
image2 = image2.reshape(1,224,224,3)

image3 = np.asarray(image3)
image3 = cv2.resize(image3, (224, 224))
image3 = image3.reshape(1,224,224,3)

yhat1 = m.predict([image1])
yhat2 = m.predict([image2])
yhat3 = m.predict([image3])

mc1 = np.argmax(yhat1[0])
prob = np.max(yhat1[0])
print(mc1)

mc2 = np.argmax(yhat2[0])
prob = np.max(yhat2[0])
print(mc2)

mc3 = np.argmax(yhat3[0])
prob = np.max(yhat3[0])
print(mc3)
