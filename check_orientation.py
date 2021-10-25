import cv2
import numpy as np

# Defining the dimensions of checkerboard
CHECKERBOARD = (7,10)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp*23

img = cv2.imread('test_afbeelding/test_dist_300mm.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+
    	cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

mtx = np.load('calibration/mtx.npy')
dist = np.load('calibration/dist.npy')
ret,rvec,tvec = cv2.solvePnP(objp, corners2, mtx, dist)
rotMat,_ = cv2.Rodrigues(rvec)

print(rotMat)
print(tvec)