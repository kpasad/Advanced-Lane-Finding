
#Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a threshold binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./op_images/chess_board_corners.png "Undistorted"
[image1a]: ./op_images/undist.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image_bin_thresh_1]:./op_images/figure_1.png "Binary1"
[image_bin_thresh_2]:./op_images/figure_2.png "Binary2"
[image_pers_1]:./op_images/figure_4.png "Perpective wrap"
[image_pers_2]:./op_images/figure_5.png "Perpective wrap"

[image_fit_1]:./op_images/figure_3-1.png "Fit lanes"
[image_fit_2]:./op_images/figure_4-1.png "Fit Lanes"
[image_superimp]:./op_images/figure_10.png "Fit Lanes"

[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"


###Camera Calibration
A camera matrix maps the 3D real world into the camera image plane.The camera calibration algorithm computes the camera matrix in terms of extrinsic and intrinsic parameters. A good explanation is found at: https://www.mathworks.com/help/vision/ug/camera-calibration.html

Camera distortion occurs due to lens imperfection. It is represented as a shift in the (x,y) position of a
pixel from its ideal location. Distortion is represented as a matrix that transforms the positional vector (x,y). 

The camera calibration function in openCV returns both the camera matrix and the distortion coefficients.
Both are used in the openCV un-distort function to correct for distortion.

###Computing the calibration and distortion matrix:

The code for this step is contained in the file calibrate.py. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained the following result. The first image shows chessboard images, overlayed with the corners detected by the cv2.findChessBoard() routine. 


A closer look at the top left images shows the bending of the top of the image (radial distortion) is un-distorted in the respective un-distorted image. A similar observation can be made about the bottom part of the top-right corner image.

It is interesting to note that the `objpoints` do not have a unit, whereas `imgpoints` are measured in pixels. This is OK since the measurement of calibration and distortion matrices depend on relative distance between the pixels rather than any absolute scale.
###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
Once the calibration and the distortion matrix is found, it does not need to be recalculated for the same camera. We store it in file. As an example, we reload the stored matrix and apply it to one of the test images. 


####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
Lane identification is preceded by image pre-processing who purpose it to remove the information elements that are not essential to or detrimental to accurate lane finding. This pre-processing occurs as a series of steps:

1. Gaussian blurring
2. Edge detection
	* RBG to alternate color representation
	* Applying Sobel edge detection to different elements of the image
	* Combining the edges dected by various methods
4.  Undistortion using the pre-calculated distortion and calibration matrix 
5.  Perspective transformation
     

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  A couple of examples of binary thresholding are below
![alt text][image_bin_thresh_1]
![alt text][image_bin_thresh_2]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

After binary image conversion, we perform the un-distortion by reusing the camera distortion matrix calculated beforehand. A perspective transformation converts the image into a bird-eye view. Perspective transformation requires the source point locations in pixels and the destination points (in pixels) where the source points are maped. Based on this information the openCV functioncv2.getPerspectiveTransform()computes a transformation matrix. The cv2.warpPerspective() then applies this transformation to the un-distroted image. See the function corners_unwrap(). 
We choose the src and destination  point as below. Other combinations would have worked fine as well.

We use 2 kinds of gradient thresholds:
Along the X axis.
Directional gradient with thresholds of 30 and 90 degrees.
This is done since the lane lines are more or less vertical.
We then apply the following color thresholds:
R & G channel thresholds so that yellow lanes are detected well.
L channel threshold so that we don't take into account edges generated due to shadows.
S channel threshold since it does a good job of separating out white & yellow lanes.

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 200, 0        | 
| 150, 720      | 200, 720      |
| 1150, 720     | 1080, 720      |
| 710, 460      | 1080, 0        |

Images below show the effect of perspective transformation.
![alt text][image_pers_1]
![alt text][image_pers_2]

####Lane Identification : The Stacked windowed approach
The general ideal is, for each lane find a sequence of connected windows along the Y-axis which has the maximum density of pixels. 

We begin at the bottom of the image, at either the location of lanes in previous frame, or absent that information, perform a histogram search across the x-axis. In the function find_lanes(), we take a histogram, and look for two maximum values (corresponding to maximum pixel density). 
A sliding window search is performed, starting with these likely positions of the 2 lanes, calculated from the histogram. Window size is uniform, and depends on the number of windows and number of vertical pixels. The windows are stacked vertically and their search reach is bounded by a parameter called `margin`. It is set to 50 pixels. I have used 10 windows of width 100 pixels.
find_lanes() returns the x and y location of the pixels.

In the parent function fit_lanes(), a second order polynomial is fit on the x and y coordinates of each lane found by the find_lanes(). Following are examples showing binary image and lanes fitted on the perceptive transformed image. 

![alt text][image_fit_1]
![alt text][image_fit_2]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The function find_curvature() computes the curvature.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
The lanes are fitted on the perspective transformed image. To display the lanes on unwraped image, the fit lanes must be un-wraped as well. The function draw_poly() first creates an image with only the lanes. It then applys a perspective transform, but with a camera matrix that is created by inverting the order of source and destination images in the perspectiveTransform() computation function. See the corners_unwrap() function where both the transformation matrices are computed.
The result of superimposing the fit lanes on baseline image are the following 

![alt text][image_superimp]

---

Lane detection in streaming video

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

A lot of time was spend fine tuning the parameters. The parameter set is generally not robust and does not stand the test of changing road conditions. An emulated data set for varying road conditions can possibly be used to build robust adaptive algorithms.
  
