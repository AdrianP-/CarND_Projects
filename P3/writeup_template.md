#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python3 drive.py model_v3.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture

The model architecture implemented is the NVIDIA Architecture, 4 convolution layers ended with a 4 Fully-connected layers. More info:  http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

However, before the model:
* Cropped the image, that action delete not interesting info in the images.
* Normalization of each pixel. 

###Training Strategy

The key is the data augmentation module:
 
```
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
```

The problem was with big curves because the prediction were too smooth, so I multiply the data with high steering angles.           
    


