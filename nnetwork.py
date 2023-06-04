import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import tensorboard as tb

import tensorflow as tf

#import tensorflow.keras
import sklearn.model_selection
#import tensorflow.keras.utils 

from tensorflow import keras
from keras import models
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import Callback
#from tensorboard import tb_callbacks
from keras.optimizers import Adam



from landmarks import mp_holistic,mediapipe_detection , draw_landmarks,draw_styled_landmarks,extract_keypoints
#Preprocess Data and Create Labels and Features
from sklearn.model_selection import train_test_split
#from tensorflow.keras.utils import to_categorical
DATA_PATH = os.path.join('MP_Data') 
no_sequences = 30
sequence_length = 30
actions = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','R','S','T','U','V','W','X','Y','Z','Hello','Help Please','Yes'])
label_map = {label:num for num, label in enumerate(actions)}


sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

#BUILD LSTM 
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu')) #activation fn
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

res = [0.7, 0.2, 0.1]

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
model.fit(X_train,y_train,epochs=30,validation_data=(X_test,y_test))
model.summary()

res = model.predict(X_test)
#!mkdir -p saved_model
model.save('saved_model/my_model')

