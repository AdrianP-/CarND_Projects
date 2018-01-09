## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

This is the last project of Udacity Self-Driving Car Nanodegree program. 
The main goal is identify vehicles from a video, also detect Lane Line as in project.

You can see the result of this project in the next link: https://youtu.be/WwBSDV8TG8A

[//]: # (Image References)
[image1]: hog_features_car.png
[image2]: hog_features_not_car.png
[image3]: slide_window.png
[image4]: slide_window_w_optimization.png
[image5]: heat_map.png
[image6]: pipeline.png

### Histogram of Oriented Gradients (HOG)

#### 1. Image features

First, I load all dataset (cars and not cars) in the section Data Loading.
Then, the function `get_hog_features` (called in `img_features` ) use the library skimage to extract hog features. 
Furthermore I extract more features like color histogram and the color spaces.
You can see the result in the next image for car and not car:
 
 Car: 
 ![alt text][image1]
 
 Not Car: 
 ![alt text][image2]

#### 2. Parameters

I tried multiple combinations of parameters and color spaces, finally based on result of SVM I use: 

```
color_space = 'LUV' 
orient = 9  
pix_per_cell = 8 
cell_per_block = 2 
hog_channel = "ALL"  
spatial_size = (32, 32) 
hist_bins = 32   
spatial_feat = True 
hist_feat = True 
hog_feat = True 
```

With this parameters I've reached and amazing 0.9928 of accuracy

#### 3. SVN train 

I use the `train_test_split` function to split the dataset in a 80% (train) - 20% (test). 
The dataset contains all features of cars and not cars.

With the 80% I trained the `LinearSVC()` and get the accuracy of prediction with the rest

### Sliding Window Search

I implemented the basic algorithm in the same way that Udacity provides. The code iterate aloTng the image subsampling in small windows. 
With those windows, predict if there are a car or not

 ![alt text][image3]
 
Then I tried the optimization in slide window as course present. I had good results also:

 ![alt text][image4]


### False Positives and Multiple detections filter

The sliding window strategy has a basic problem, the multiple detections. 
The same car could appeard in different windows, so the SVC classifier will predict True but it will be the same car.
Besides we want to remove false positives, to address this, I used the heat map strategy, adding +1 for all pixels within where a positive detection

Example:

 ![alt text][image5]
 
### Pipeline: 

To make the final video, is necessary gather all project (also project 4). I defined the next global to call in each frame of video:

```
import lane_finding

def vehicle_detect_and_line_lines(image):
    image = np.copy(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    _ , boxes = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    draw_img, _ = filter_boxes(lane_finding.process_image(image), boxes)
     
    return cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
```
From image to lane finding and car position:

 ![alt text][image6]

---

### Discussion

In spite of the good accuracy, the real world is too complex, so always will have false positve and false negatives. 
We should use a new frameworks and strategies (e.g. DNN, YOLO etc) at the same time (throw a ensemble), but not only this,
 we should improve the sensors, cameras, etc etc
 
Other point to improve is the speed of execution, the whole pipeline should have a result in less than 100ms, far away from this project