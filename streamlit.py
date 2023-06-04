import streamlit as st
import os 
import imageio
import cv2 

import tensorflow as tf 
# from utils import load_data, num_to_char
# from modelutil import load_model
#from predictions import words
words= "Help Please "
# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('ASL Translation')
    st.info('This application is originally developed for Synapse Company .')

st.title('Sign Language Translation App') 
#st.title('Sign Language Translation App')
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
else:
    st.write('Stopped')
if st.button('Translate'):

    st.write(words) #displayed when the button is clicked

else:

    st.write(" ") #displayed when the button is unclicked
