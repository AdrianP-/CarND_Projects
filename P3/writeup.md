#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---
### Files Submitted & Code Quality

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### Functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python3 drive.py model_v3.h5
```

### Data set generation
To make the dataset, I recorded one lap using the mouse. Besides I took care driving always the car in the middle of the road. 
Also I used the flipped technique of data augmentation, because before that, my car only learned to turn left.


The key is the data augmentation module:
 
```
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
```

The problem was with big curves because the angle prediction was very smooth, so I multiply the data with high steering angles and increment the angle by 20%.
Besides, the dataset it was really unbalanced (there are a lot of zero angles), so only added the 30% of zero angles (in random way) 


In summarize: 
* Aligning color spaces (RGB, not BGR)
* Applied data agumentations techniques
* Delete unbalanced data. 

### Model Architecture

The model architecture implemented is the same as NVIDIA Architecture, 4 convolution layers ended with a 4 Fully-connected layers. More info:  http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

However, before the model:
* Cropped the image, that action delete not interesting info in the images.
* Normalize each pixel


### KISS (Keep It Simple)

I didn't add multiple cameras because I thought that is not necessary to solve this problem :)

### Finally
Because this is a problem of regression, I used mse metric. 
Next I shuffled the data set and split it in a 90%-10% distribution (train/test). 5 epochs was enough because with more epochs the NN began to overfit.  

### Results
Thank to all those techniques the car can drive by itself without pop out onto ledges or roll over surfaces in whole lap. 
