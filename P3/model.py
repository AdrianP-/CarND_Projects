import csv
import random

import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn

current_path = "/home/adrian/carnd/CarND-Behavioral-Cloning-P3"
os.chdir(current_path)
data = pd.read_csv("data/driving_log.csv", header=None)


def factor_reduce(value):
    return int(value / 1)

# def generator(samples, batch_size=32):
#     num_samples = len(samples)
#     while 1: # Loop forever so the generator never terminates
#         random.shuffle(samples)
#         for offset in range(0, num_samples, batch_size):
#             batch_samples = samples[offset:offset+batch_size]
#
#             images = []
#             angles = []
#             for batch_sample in batch_samples:
#                 name = './IMG/'+batch_sample[0].split('/')[-1]
#                 center_image = cv2.imread(name)
#                 center_angle = float(batch_sample[3])
#                 images.append(center_image)
#                 angles.append(center_angle)
#
#             # trim image to only see section with road
#             X_train = np.array(images)
#             y_train = np.array(angles)
#             yield sklearn.utils.shuffle(X_train, y_train)

images = []
measurements = []
for idx, line in data.iterrows():
    image = cv2.imread(line[0])
    new_shape = (factor_reduce(image.shape[1]), factor_reduce(image.shape[0]))
    image = cv2.resize(image, new_shape)
    images.append(image)
    steering = float(line[3])
    measurements.append(steering)

augmented_images, augmented_measurementes = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurementes.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurementes.append(measurement * -1.0)
    if measurement >= 0.5 or measurement <= -0.5:
        for i in range(5):
            augmented_images.append(image)
            augmented_measurementes.append(measurement)
            augmented_images.append(cv2.flip(image, 1))
            augmented_measurementes.append(measurement * -1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurementes)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Cropping2D, K
from keras.layers import Convolution2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(factor_reduce(160), factor_reduce(320), 3)))
model.add(Cropping2D(cropping=((factor_reduce(70), factor_reduce(25)), (0, 0))))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.1, shuffle=True, nb_epoch=5, verbose=1)

# ### print the keys contained in the history object
# print(history_object.history.keys())
#
# ### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()

model.save('model_v3.h5')
