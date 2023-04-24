import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16

from sklearn.model_selection import train_test_split
from keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


videos_dir = 'videos'
output_dir = 'images'
image_size = (224, 224)

CLASSES_LIST = ['Sitting', 'Sleeping', 'Waving']

X = []
y = []


# extracting frames from videos
for folder_name in os.listdir(videos_dir):
    folder_path = os.path.join(videos_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue
    
    output_folder_path = os.path.join(output_dir, folder_name)
    os.makedirs(output_folder_path, exist_ok=True)

    for video_name in os.listdir(folder_path):
        if not video_name.endswith('.mp4'):
            continue
        
        video_path = os.path.join(folder_path, video_name)

        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)

        count = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            if count % int(fps/2) == 0:
                frame_resized = cv2.resize(frame, image_size)

                output_image_name = f"{os.path.splitext(video_name)[0]}_{count}.jpg"
                output_image_path = os.path.join(output_folder_path, output_image_name)
                cv2.imwrite(output_image_path, frame_resized)
            count += 1

        cap.release()


# creating dataset
for c in os.listdir(output_dir):
    path = os.path.join(output_dir, c)
    for i in os.listdir(path):
        p = os.path.join(path, i)
        im = cv2.imread(p)
        X.append(im)
        y.append(CLASSES_LIST.index(c))



X = np.asarray(X)
y = np.asarray(y)
# normalizing
X = X / 255

# one hot encoding
one_hot_encoded_labels = to_categorical(y,3)
print(one_hot_encoded_labels)
features_train, features_test, labels_train, labels_test = train_test_split(X, one_hot_encoded_labels,
                                                                            test_size = 0.25, shuffle = True,
                                                                            random_state = 10)


# model

def create_model(input_shape, optimizer='adam', fine_tune=0):
    # Model's fully-connected layers are frozen
    conv_base = VGG16(include_top=False,
                     weights='imagenet', 
                     input_shape=input_shape,
                     )
    
    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    # Bootstrapping a new top_model onto the pretrained layers
    top_model = conv_base.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(10, activation='relu')(top_model)
    top_model = Dense(10, activation='relu')(top_model)
    top_model = Dropout(0.3)(top_model)
    output_layer = Dense(3, activation='softmax')(top_model)
    
    # Group the convolutional base and new fully-connected layers into a Model object.
    model = Model(inputs=conv_base.input, outputs=output_layer)

    # Compiling
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


vgg_model = create_model((224,224,3), optimizer='adam', fine_tune=0)
print("Model successfully created")
early_stopping_callback = EarlyStopping(monitor = 'val_loss', verbose=1,patience=2)
vgg_model.fit(x = features_train, y = labels_train, epochs = 10, batch_size = 10,
                                                     shuffle = True, validation_split = 0.2, 
                                                     callbacks = [early_stopping_callback])
vgg_model.evaluate(features_test, labels_test)


vgg_model.save('rsp4.h5')


import matplotlib.pyplot as plt

y_pred = vgg_model.predict(features_test)
y_pred_classes = np.argmax(y_pred, axis=1)
true_one_hot = np.argmax(labels_test, axis=1)

# confusion matrix
from sklearn.metrics import confusion_matrix
confmat=confusion_matrix(true_one_hot, y_pred_classes)
print(confmat)


# percision and recall
from sklearn.metrics import precision_score, recall_score
precision = precision_score(true_one_hot, y_pred_classes, average='weighted')
recall = recall_score(true_one_hot, y_pred_classes, average='weighted')

print("Precision:",precision)
print("Recall:",recall)


# Plotting Training Loss and validation loss    
train_loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = np.arange(len(train_loss)) + 1

plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
