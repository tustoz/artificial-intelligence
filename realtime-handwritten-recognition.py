# Import necessary libraries
import cv2
import numpy as np

# Load the trained model
model = load_model('handwriting_recognition_model.h5')

# Create a function for preprocessing the input handwriting
def preprocess_handwriting(img):
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to the image
    img_threshold = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)[1]
    
    # Apply edge detection to the image
    img_edges = cv2.Canny(img_threshold, 50, 150)
    
    # Return the preprocessed image
    return img_edges

# Create a function for predicting the text in the input handwriting
def predict_handwriting(img):
    # Preprocess the input image
    img_preprocessed = preprocess_handwriting(img)
    
    # Resize the image to match the input size of the model
    img_resized = cv2.resize(img_preprocessed, (28, 28))
    
    # Convert the image to a 4D tensor for input to the model
    img_input = img_resized.reshape(1, 28, 28, 1)
    
    # Use the model to predict the text in the input handwriting
    prediction = model.predict(img_input)
    
    # Return the predicted text
    return prediction

# Set up the video capture
cap = cv2.VideoCapture(0)

# Loop until the user quits
while True:
    # Capture the frame from the video
    ret, frame = cap.read()
    
    # Check if the frame was captured successfully
    if ret == True:
        # Predict the text in the frame
        prediction = predict_handwriting(frame)
        
        # Display the predicted text on the screen
        cv2.putText(frame, prediction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the frame on the screen
        cv2.imshow('Realtime Handwriting Recognition', frame)
        
        # Check if the user pressed the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
