# Vehicle Detection - P5
## Self-Driving Car Nanodegree

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4), but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

[//]: # (Image References)
[image1]: ./output_images/car_hog.png
[image2]: ./output_images/notcar_hog.png
[image3]: ./output_images/box_overlay.png
[image4]: ./output_images/heatmap.png

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

extract_features() is the function that is used to extract features from a list of images. It calls call bin_spatial() and color_hist()
bin_spatial() is used to resize the image to 32x32 using cv2.resize and create features vector. it takes a bin_spatial(img, size=(32, 32)): and returns a return np.hstack((features0,features1,features2)). def color_hist(img, nbins=32) takes in image and returns returns indivuaial histograms and bincenters as concatenated values. 

the same slimdown version for illustrative purpose is single_img_features() which takes in single image instead of multiple images. it takes in (image, colorspace "rgb" or "hsv" or "luv" or "hls" or "yuv" or "YCrCb", spatial size which is 32*32, orientation is 9, pixel per cell is 8 with 2 cells per block and 32 histogram bins 

the same input parametrs is used in extract_features example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image1]
![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of `skimage.hog()` and the best performance of HOG parametes was with `pixels_per_cell=(8, 8)`
 `orientations=9`,  and `cells_per_block=(2, 2)` with `YCrCb` color space seems that best. 
 
 http://waset.org/publications/10001883/a-background-subtraction-based-moving-object-detection-around-the-host-vehicle this paper too gives a better explanation on it.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The 10th cell in the ipynb i use a linear SVC to train the classifier which gave a stunning result of 0.9932% accuracy which is pretty decent. After extracting the HOG features with the colorspace and above explained parameters, combined with spatially binned color and histograms of color. Then I stack these features into a single numpy array. To avoid overfitting I split up data into randomized training and test sets with sklearn `train_test_split`.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

First of all, I only apply the sliding window on the bottom half of the image because the vehicles will not show above the road surface it was an excllent clue from the lesson which could have been easily overlooked. I tried different window size (search scale) and overlap rate, and found that window size 96 and overlap 0.5 works well. 

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

![alt text][image3]
I tried various combinations of YCrCb ALL-channel HOG features,spatially binned color and histograms of color. i varied the output with diffrent color space, and the HOG parameter.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

[![IMAGE ALT TEXT](http://img.youtube.com/vi/43ia0BKZJjY/0.jpg)](https://www.youtube.com/watch?v=43ia0BKZJjY "Test Video ")

[![IMAGE ALT TEXT](http://img.youtube.com/vi/I0Uvo9OyVoc/0.jpg)](https://www.youtube.com/watch?v=I0Uvo9OyVoc "Result Video ")


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  we assumed each blob corresponded to a vehicle, and finally we draw bounding boxes to cover the area of each blob detected. The heat map is generated from the positions of positive detected cars from the trained classifer and by summing up and recorded the heatmap for 10 sequential frame and apply a threshold of 5. We get a new heatmap based on last generated positions

### Here are six frames and their corresponding heatmaps:

![alt text][image4]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

To my own intrest i tried in a sample from a car recorded from Chennai, India. The issue was the model was trained only to identify cars it drastically failed in identifying 2-wheelers (0:29 sec) and pedestrians which i felt could be added to this project to make it a complete one. <br>
There were lot of false positve in the video from (0:31) to (0:40) which might inturn in a real-time make a steering correction and might result in unwanted breaking and turning. <br>
To my surprise the white car at (1:45) wasnt identifed which was right in front most of the time. Cars very close to the dashcam failed to identify <br>

[![IMAGE ALT TEXT](http://img.youtube.com/vi/ntggVyuKGsc/0.jpg)](https://www.youtube.com/watch?v=ntggVyuKGsc "Chennai Video ")

To optimize the algorithm i felt we already know which lane the car is in using previous projects. we can use this algo only for calcualting cars in the adjacent lanes only. instead of the whole frame. <br>

The algorithm will fail in a downhill traffic. where the opposite side car might be in the top left or right of the windshield. we might not even consider that case in this algorithm.<br>

The algo might also suffer when there is a bridge a short road (flat) and another bridge. when our car is in the bridge (downstream) and if there is any car in the short road that will not be seen. similarly when our car is in flatroad and if there is another car climbing that might also be missed <br>

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation. !
