import cv2
import dlib
import torch

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the facial landmark detector
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the pre-trained sentiment analysis model
model = torch.load('sentiment_analysis_model.pt')

# Start the video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # For each face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Detect facial landmarks
        landmarks = predictor(gray, dlib.rectangle(x, y, x+w, y+h))

        # Generate a caption describing the emotions
        caption = generate_caption(landmarks)

        # Use the sentiment analysis model to predict the sentiment of the caption
        sentiment = model(caption)

        # Draw the caption and sentiment on the frame
        cv2.putText(frame, caption, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, sentiment, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
video_capture.release()
