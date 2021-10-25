from ctypes import RTLD_GLOBAL
import cv2
import numpy as np
import glob
import math

# Defining the dimensions of checkerboard
CHECKERBOARD = (7,10)
check_size = 23
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp*check_size
prev_img_shape = None

def set_calibration():
    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [] 

    # Extracting path of individual image stored in a given directory
    images = glob.glob('./afbeeldingen/*.jpg')
    for fname in images:
        print('processing image:' + fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+
            cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2,ret)
        
    cv2.destroyAllWindows()

    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    np.save('calibration/mtx.npy',mtx)
    np.save('calibration/dist.npy',dist)

    """
    print("Camera matrix : \n")
    print(mtx)
    print("dist : \n")
    print(dist)
    print("rvecs : \n")
    print(rvecs)
    print("tvecs : \n")
    print(tvecs)
    """

def calibrate_robot(image):
    #frame = get_camera_image()
    #if(type(frame) == bool):
    #    return False

    #cv2.imshow('frame', frame)
    #cv2.waitKey(0)

    img = cv2.imread(image)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+
            cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not (ret):
        raise ValueError()

    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

    mtx = np.load('calibration/mtx.npy')
    dist = np.load('calibration/dist.npy')
    ret,rvec,tvec = cv2.solvePnP(objp, corners2, mtx, dist)
    rotMat,_ = cv2.Rodrigues(rvec)



    return (rotMat, tvec)

def get_euler_angles (rotMatBase, rotMatCal):
    rotMatTot = np.matmul(rotMatBase.transpose(),rotMatCal)
    rx = math.atan(rotMatTot[2,1]/rotMatTot[2,2])
    ry = - math.sin(rotMatTot[2,0])
    rz = math.atan(rotMatTot[1,0]/rotMatTot[0,0])

    rx = rx*180/math.pi
    ry = ry*180/math.pi
    rz = rz*180/math.pi
    
    print ('rx: {rx}'.format(rx = rx))
    print ('ry: {ry}'.format(ry = ry))
    print ('rz: {rz}'.format(rz = rz))

    return (rx, ry, rz)


def get_camera_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if(ret):
        return frame
    else:
        raise ValueError()

if __name__ == "__main__":
    (rotMat_1, tvec) = calibrate_robot('test_afbeelding/test_dist_300mm.jpg')
    #print(rotMat_1)
    #print(tvec)
    (rotMat_2, tvec) = calibrate_robot('test_afbeelding/links_onder.jpg')
    #print(rotMat_2)
    #print(tvec)
    get_euler_angles(rotMat_1,rotMat_2)
    
    