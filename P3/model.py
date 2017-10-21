import csv
import random

import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

'''
In case that I need image aumentation module
'''
# def image_augmentation(images):
#     # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
#     # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
#     sometimes = lambda aug: iaa.Sometimes(1, aug)
#
#     # Define our sequence of augmentation steps that will be applied to every image
#     # All augmenters with per_channel=0.5 will sample one value _per image_
#     # in 50% of all cases. In all other cases they will sample new values
#     # _per channel_.
#     seq = iaa.Sequential(
#         [
#             iaa.SomeOf((0, 5),
#                 [
#                     iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
#                     iaa.Add((-15, 15), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
#                     iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
#                 ],
#                 random_order=True
#             )
#         ],
#         random_order=True
#     )
#
#     return seq.augment_images(images)

#Get Dataset
current_path = "/home/adrian/carnd/CarND-Behavioral-Cloning-P3"
os.chdir(current_path)
data = pd.read_csv("data/driving_log.csv", header=None)

'''
Finally I didn't use generator because the dataset fits perfectly on my computer
'''
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

#Get the images (features) and theirs steering angles (labels)
images = []
measurements = []
for idx, line in data.iterrows():
    image = cv2.imread(line[0])
    b, g, r = cv2.split(image) #switch to RGB image
    image = cv2.merge([r, g, b])
    # new_shape = (image.shape[1], image.shape[0])
    # image = cv2.resize(image, new_shape)
    images.append(image)
    steering = float(line[3])*1.20 #angle aumentation by 20%
    measurements.append(steering)


def decision_to_add(probability=0.7):
    return random.random() > probability



'''
Data augmentation module:
    1- Flip each image
    2- multiplied the images with big steering angles
'''
augmented_images, augmented_measurements = [], []


for image, measurement in zip(images, measurements):
    print(measurement, measurement == 0, decision_to_add())
    if measurement == 0 and decision_to_add():
        augmented_images.append(image)
        augmented_measurements.append(measurement)
    else:
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image, 1))
        augmented_measurements.append(measurement * -1.0)

    if measurement >= 0.4 or measurement <= -0.4:
        for i in range(5):
            augmented_images.append(image)
            augmented_measurements.append(measurement)
            augmented_images.append(cv2.flip(image, 1))
            augmented_measurements.append(measurement * -1.0)

# print(augmented_measurements)
# import matplotlib.pyplot as plt
# plt.hist(augmented_measurements, bins='scott')  # arguments are passed to np.histogram
# plt.title("Histogram with 'auto' bins")
# plt.show()



X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


'''
Nvidia architecture for deep learning with normalization and cropping in first layers
'''
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Cropping2D, K, Dropout
from keras.layers import Convolution2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

#Traning
history_object = model.fit(X_train, y_train, validation_split=0.1, shuffle=True, nb_epoch=7, verbose=1)

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






