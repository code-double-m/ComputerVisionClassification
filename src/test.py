from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('cat_recognition_model.h5')

# Load an example image for prediction
img_path = 'black.jpg'
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize the pixel values to be between 0 and 1

# Make a prediction
prediction = model.predict(img_array)

# Check the prediction result
if prediction[0][0] > 0.5:
    print("The image contains a cat!")
else:
    print("The image does not contain a cat.")

print(prediction[0][0])
