
import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import ImageGrid


def img_grid(image,title):
    n = int(np.ceil(np.sqrt(len(image))))
    fig = plt.figure(1, (40., 40.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(n, n),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
       
    for i in range(len(image)):
        #grid[i].imshow(image[i],cmap='gray',animated=True)  # The AxesGrid object work as a list of axes.
        grid[i].imshow(image[i],cmap='gray')  # The AxesGrid object work as a list of axes.
        #grid[i].set_title(title[i])


# prepare object points
nx = 9 # The number of inside corners in x
ny = 6 # The number of inside corners in y

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

list_of_images=[]
title = [None]*len(list_of_images)

# Step through the list and search for chessboard corners
for i in range(1,21):
    fname = 'camera_cal/calibration{}.jpg'.format(i)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

    # If found, add object points, image points
    if ret == True:
        print(fname)
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (8,6), corners, ret)
        list_of_images.append(img)
        
#img_grid(list_of_images[0:4],title[1:4])    

list_of_images=[]
title = [None]*len(list_of_images)

img_size = (img.shape[1], img.shape[0])
# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
for i in [2,3,6,7,8,9,10]:
    fname = 'camera_cal/calibration{}.jpg'.format(i)
    print(fname)

    img = cv2.imread(fname)
    img_size = (img.shape[1], img.shape[0])

    dst = cv2.undistort(img, mtx, dist, None, mtx)
    list_of_images.append(dst)

pickle.dump( [mtx,dist], open( "./dist_params.p", "wb" ) )

#img_grid(list_of_images[0:4],title[1:4])        
# Test undistortion on an image
fname = 'test_images/test1.jpg'
img = cv2.imread(fname)
img_size = (img.shape[1], img.shape[0])


dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imshow('img',dst)
#cv2.imwrite('calibration_wide/test_undist.jpg',dst)

# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=15)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=15)