import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('cat_recognition_model.h5')

# Open a connection to the webcam (0 indicates the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize the frame to the input size of the model
    resized_frame = cv2.resize(frame, (64, 64))

    # Convert the frame to a format suitable for prediction
    img_array = image.img_to_array(resized_frame)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make a prediction
    prediction = model.predict(img_array)

    # Display the original frame
    cv2.imshow('Cat Recognition', frame)

    # Check the prediction result and display it
    if prediction[0][0] > 0.5:
        cv2.putText(frame, 'Cat Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'No Cat Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the processed frame with the prediction result
    cv2.imshow('Cat Recognition', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
