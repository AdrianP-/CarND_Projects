## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions 
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 


Writeup
---
My principal points to reach +0.93 accuracy was: 
* Image aumentation: The dataset was unbalanced (more data in some labels than others). So I did a piece of code that balance each label. To create new bunch of images I used imgaug library (in image_augmentation function). Thanks to that it wasn't necessary applied a normalization.  

* Dropout in LeNet: In the first version, the NN overfitting too fast, so I added a dropout layer `conv2 = tf.nn.dropout(conv2, keep_prob)`   
 
Instead of download new images, I get random samples of X_test data. The prediction and performace was quite good. 

Finally, in softmax predictions I took the `display_image_predictions` function from my second project of DeepLearning Nanodegree. 


#### Architecture
As I mencioned, I use a LeNet network. Configuration:
* 5 layers
    * 1-2: Convolutionals
    * 3-5: Fully connected
* relu activations
* dropout in the end of layer 2
* Input NN shape = 32x32x5
* Output NN: number of different labels


#### Hyperparameters
`EPOCHS = 100` -> With more epochs the NN starts to overfitting

`BATCH_SIZE = 64` -> Good results in terms of generalization 

`rate = 0.0005` -> With more learning rate I had huge jumps in performance

#### Test model
Once the NN was trained, It will be used to predict new images that it never saw. 
I randomly choose 5 images from test dataset and predict it. The results was quite good except in too much blurred images or dark. However thanks to add 3 channels of colors (instead of grayscale) the NN can predict better the darkest images.     

Also the accuracy is quite good but always worse than the accuracy performance (obviously)
 
### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

### Requirements for Submission
Follow the instructions in the `Traffic_Sign_Classifier.ipynb` notebook and write the project report using the writeup template as a guide, `writeup_template.md`. Submit the project code and writeup document.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

