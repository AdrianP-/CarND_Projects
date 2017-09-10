# **Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

Overview
---

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project you will detect lane lines in images using Python and OpenCV.  OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.  

The writeup
---

The pipeline is base on the last Quiz

1- transform de image to grayscale

2- apply the gaussian blur

3- Filter edges with high values

4- Select the region of interest. I make a triangle with highest dot in the middle-top
  
5- And finally apply Hough function and wighted img

```
def pipeline(image):
    gray = grayscale(image)
    
    kernel_size = 5
    gausian = gaussian_blur(gray, kernel_size)
    low_threshold = 260
    high_threshold = 300
    edges = canny(gausian,low_threshold, high_threshold)
    
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]), (imshape[1],imshape[0]), (imshape[1]/2, imshape[0]/2)]], dtype=np.int32)

    masked_edges = region_of_interest(edges, vertices)
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 30     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 10  #minimum number of pixels making up a line
    max_line_gap = 4     # maximum gap in pixels between connectable line segments
        
    lines = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    
    color_edges = np.dstack((edges, edges, edges))
    lines_edges = weighted_img(img=lines, initial_img=color_edges)
    
    return lines_edges 
```
Improvements: I select the region of interest, and works well in the image, but seems that in the video doesn't work correctly
