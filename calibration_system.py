import cv2
import numpy as np
import glob
import math

# Defining the dimensions of checkerboard
CHECKERBOARD = (7,10)   # (heigth,width)
check_size = 23     # size of squares in mm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Defining the coördinates of the checkerboard crosspoints that the camera will detect
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp*check_size

def calibrate_camera():
    # Creating vector to store coordinates checkerboard crosspoints in real world
    objpoints = []
    # Creating vector to store coordinates checkerboard crosspoints on image
    imgpoints = [] 

    # Define path for images stored in a given directory
    images = glob.glob('./afbeeldingen/*.jpg')

    for fname in images:
        #print('processing image:' + fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board crosspoints
        # If desired number of crosspoints are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+
            cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        """
        If desired number of crosspoints are detected,
        pixel coordinates are refined and coordinates are stored in vector
        """
        if ret == True:
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            
            # Add coordinate points to vector
            imgpoints.append(corners2)  # coordinates in image
            objpoints.append(objp)      # coordinates in real world
        

    """
    Performing camera calibration by 
    passing the value of known coordinates in real world (objpoints)
    and corresponding pixel coordinates in images of the 
    detected crosspoints (imgpoints)
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    np.save('calibration/mtx.npy',mtx)
    np.save('calibration/dist.npy',dist)

    #print("Camera matrix : \n")
    #print(mtx)
    #print("dist : \n")
    #print(dist)
    #print("rvecs : \n")
    #print(rvecs)
    #print("tvecs : \n")
    #print(tvecs)

def _get_rot_dist(image):
    #frame = get_camera_image()
    #if(type(frame) == bool):
    #    return False

    #cv2.imshow('frame', frame)
    #cv2.waitKey(0)

    # Get grayscale of image and find crosspoints
    img = cv2.imread(image)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+
            cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not (ret):
        raise ValueError()

    # Refine pixel coordinates
    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

    # Load intrinsic variables of camera
    mtx = np.load('calibration/mtx.npy')
    dist = np.load('calibration/dist.npy')

    # Calculate distance vector and rotation matrix
    ret,rvec,tvec = cv2.solvePnP(objp, corners2, mtx, dist)
    rotMat,_ = cv2.Rodrigues(rvec)

    return (rotMat, tvec)

def _get_camera_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if(ret):
        return frame
    else:
        raise ValueError()

def initial_calibration():
    (calRotMat, calTvec) = _get_rot_dist('test_afbeelding/test_dist_300mm.jpg')
    np.save('calibration/calRotMat.npy',calRotMat)
    np.save('calibration/calTvec.npy',calTvec)

def operation_calibration():
    calRotMat = np.load('calibration/calRotMat.npy')    
    calTvec = np.load('calibration/calTvec.npy')

    (operRotMat, operTvec) = _get_rot_dist('test_afbeelding/links_onder.jpg')

    tvecRel = calTvec - operTvec
    rvecRel = _get_euler_angles(operRotMat,calRotMat)
    return [tvecRel.item(0),tvecRel.item(1),tvecRel.item(2),rvecRel[0],rvecRel[1],rvecRel[2]]
    
def _get_euler_angles (rotMatBase, rotMatCal):
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



if __name__ == "__main__":
    initial_calibration()
    changeVec = operation_calibration()
    print(changeVec)
    
    