import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator



class Trainer():
    def __init__(self, training_data_path, img_scale=64, batch_size=32, epochs=100, model_name="trainer_model.h5"):
        self.model_name = model_name
        self.epochs = epochs
        self.img_width, self.img_height = 64,64
        self.batch_size = batch_size
        self.train_generator = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True).flow_from_directory(training_data_path, target_size=(self.img_width, self.img_height),
                                                                                                                                             batch_size=self.batch_size,class_mode='binary')

        self.model = Sequential()

        self.model.add(Conv2D(32, (3, 3), input_shape=(self.img_width, self.img_height, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))


    def compile_and_train(self):
        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        self.model.fit(self.train_generator, epochs=self.epochs)

        # Save the model
        seld.model.save('cat_recognition_model.h5')



t = Trainer("training_data")
t.compile_and_train()
