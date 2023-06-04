import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import tensorflow as tf
#import tensorflow.keras
# import sklearn.model_selection
#import tensorflow.keras.utils 
# from tensorflow import keras
# from keras import models
from landmarks import mp_holistic,mediapipe_detection , draw_landmarks,draw_styled_landmarks,extract_keypoints
#from nnetwork import labels,sequences,X_test,y_test,actions,action
from nnetwork import labels,sequences,actions,action
from nnetwork import model


loaded = tf.saved_model.load('saved_model/my_model')

import tensorflow as tf
new_model = tf.keras.models.load_model('saved_model/my_model')

# Check its architecture
new_model.summary()

new_model.load_weights('saved_model/my_model')
#es = model.predict(X_test)

#from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
# model=Sequential()
# X = np.array(sequences)
# y = to_categorical(labels).astype(int)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
# yhat =new_model.predict(X_test)
# ytrue = np.argmax(y_test,axis=1).tolist()
# yhat = np.argmax(yhat,axis=1).tolist()
# accuracy_score(ytrue,yhat)

from scipy import stats
colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            
        #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def speak():
    import pyttsx3  
# initialize Text-to-speech engine 
    engine = pyttsx3.init()  
# convert this text to speech  
    text = action
    engine.say(text)  
# play the speech  
    engine.runAndWait()  
speak()