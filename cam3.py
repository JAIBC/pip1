import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('cam_model.h5')

# Define the classes
classes = ['Cigarette', 'Phone', 'Coda', 'None']

# Define the Streamlit app
st.set_page_config(page_title='Image Classification App', page_icon=':camera:', layout='wide')
st.header('Webcam Preview')

# Define the function to make predictions
def predict(image):
    # Preprocess the image
    image = cv2.resize(image, (224, 224)) # adjust the size of the image to match the model's input shape
    image = np.array(image) / 255.0 # normalize the image pixels
    image = np.expand_dims(image, axis=0) # add a batch dimension
    
    # Make a prediction
    prediction = model.predict(image)[0]
    prediction = np.argmax(prediction)
    
    # Get the class name
    class_name = classes[prediction]
    
    return class_name

# Define the function to show the webcam preview
def show_webcam():
    cap = cv2.VideoCapture(0) # open the default camera
    while True:
        ret, frame = cap.read() # read a frame
        if not ret: # break the loop if the frame cannot be read
            break
         # convert the frame to RGB format
        class_name = predict(frame) # make a prediction
        cv2.putText(frame, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2) # add the prediction to the frame
        cv2.imshow('Webcam Preview', frame) # show the frame
        if cv2.waitKey(1) == ord('q'): # break the loop if the 'q' key is pressed
            break
    cap.release() # release the camera
    cv2.destroyAllWindows()

# Start the webcam preview
show_webcam()