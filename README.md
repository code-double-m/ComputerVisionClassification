# ComputerVisionClassification

## Cat Recognition Project

This project consists of three components for recognizing cats in images using a Convolutional Neural Network (CNN). The model is trained using a custom dataset, and the following scripts handle different use cases for prediction: training a model, real-time webcam prediction, and static image prediction.

## Files

- `trainer.py` - Script for training the cat recognition model using a dataset of images.
- `cat_recognition_model.h5` - The saved trained model used for prediction.
- `cat_recognition_webcam.py` - Script for real-time cat detection using the webcam.
- `cat_recognition_image.py` - Script for predicting if a static image contains a cat.

## Requirements

To install the necessary packages for all scripts, run:

pip install tensorflow opencv-python numpy

## Dataset Format

For training, the dataset should be organized in a directory with subdirectories for each class. Example structure:




## Cat Recognition Model Trainer (`trainer.py`)

This script defines a `Trainer` class that creates, compiles, trains, and saves a convolutional neural network (CNN) model for binary image classification (cats vs. non-cats).

### How to Run

1. Ensure the `training_data` directory is organized with images.
2. Run the script:

python trainer.py

The script will:
- Load the training images using `ImageDataGenerator`
- Build a CNN model with three convolutional layers followed by max-pooling layers
- Train the model using binary cross-entropy loss and the Adam optimizer
- Save the trained model as `cat_recognition_model.h5`

### Model Architecture

- **Conv2D layers** with ReLU activation for feature extraction
- **MaxPooling2D layers** to downsample the image representations
- **Flatten layer** to reshape the output for the fully connected layers
- **Dense layers** with ReLU and sigmoid activation for classification

## Cat Recognition Using Webcam (`cat_recognition_webcam.py`)

This script uses a pre-trained CNN model (`cat_recognition_model.h5`) to detect cats in real-time from the webcam feed. It captures video frames, processes them, and uses the model to predict whether a cat is present in the frame.

### How to Run

1. Ensure the `cat_recognition_model.h5` file is in the same directory.
2. Run the script:

python cat_recognition_webcam.py

The script will:
- Open a connection to the webcam
- Continuously capture frames from the webcam
- Preprocess each frame to match the model input size
- Use the trained model to predict whether a cat is detected
- Display the webcam feed with a label showing "Cat Detected" or "No Cat Detected" based on the prediction

To exit the webcam feed, press the 'q' key.

### Model Input and Prediction

- The webcam frames are resized to 64x64 pixels to match the model's input size.
- The frames are converted to arrays and normalized (divided by 255) before being passed to the model.
- The model outputs a probability, and based on the threshold of 0.5, it classifies the frame as either "Cat Detected" or "No Cat Detected".

## Cat Recognition for Static Images (`cat_recognition_image.py`)

This script loads a pre-trained cat recognition model (`cat_recognition_model.h5`) and uses it to predict whether a given image contains a cat or not.

### How to Run

1. Ensure the `cat_recognition_model.h5` file and the image file (e.g., `black.jpg`) are in the same directory.
2. Run the script:

python cat_recognition_image.py

The script will:
- Load a specified image file (default is `black.jpg`)
- Preprocess the image to match the input size of the model (64x64 pixels)
- Normalize the image pixel values to be between 0 and 1
- Use the trained model to predict if a cat is present in the image
- Print the result, indicating whether the image contains a cat or not

### Image Input

The image is loaded and resized to 64x64 pixels to match the model's expected input size. The image is then normalized by dividing the pixel values by 255 to bring them between 0 and 1.

### Prediction

The model's output is a probability value between 0 and 1. If the value is greater than 0.5, the script prints "The image contains a cat!" Otherwise, it prints "The image does not contain a cat."

## Notes

- For all scripts, ensure that the model `cat_recognition_model.h5` is correctly trained and available.
- The training dataset should be organized according to the format mentioned in the "Dataset Format" section.
- The webcam script (`cat_recognition_webcam.py`) requires a functional webcam and access to OpenCV.
- The image prediction script (`cat_recognition_image.py`) works for static image files and can be adapted for batch processing or integration into other applications.
