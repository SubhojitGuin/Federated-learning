import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

def load_model():
    # Define the CNN model and its optimizer
    input_layer = tf.keras.Input(shape=(28, 28, 1))

    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    pool1 = keras.layers.MaxPooling2D((2, 2))(conv1)

    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu')(pool1)
    pool2 = keras.layers.MaxPooling2D((2, 2))(conv2)

    dropout1 = keras.layers.Dropout(0.25)(pool2)

    flatten = keras.layers.Flatten()(dropout1)

    dense1 = keras.layers.Dense(128, activation='relu')(flatten)
    dropout2 = keras.layers.Dropout(0.5)(dense1)

    output_layer = keras.layers.Dense(10, activation='softmax')(dropout2)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def load_dataset():
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    return X_train, y_train, X_test, y_test