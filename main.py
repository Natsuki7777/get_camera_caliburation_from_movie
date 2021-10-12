import cv2
import numpy as np
import glob
import os
import sys


def get_n_frame(
        dir_path,
        basename,
        input_n=40,
        ext='jpg',
):
    video_list = glob.glob(r'.\video\*')
    for video_path in video_list:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return

        os.makedirs(dir_path, exist_ok=True)
        base_path = os.path.join(dir_path, basename)

        digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if input_n == 0:
            dur = 1
        else:
            dur = frame_count // input_n

        frame_num_list = np.array(range(10, frame_count - 10, dur))

        for frame_num in frame_num_list:

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

            ret, frame = cap.read()

            if ret:
                cv2.imwrite(
                    '{}_{}.{}'.format(
                        base_path, str(frame_num).zfill(digit), ext
                    ),
                    frame
                )


def get_camera_caliburation():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7 * 10, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:10].T.reshape(-1, 2)

    images = glob.glob(r'.\frames\*.jpg')

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 10), None)

        # If found, add object points, image points (after refining them)
        if ret:
            print(ret, fname)
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            img = cv2.drawChessboardCorners(img, (7, 10), corners2, ret)
            cv2.imwrite("{}\\{}".format('chessboardcorners', fname[9:]), img)
            cv2.waitKey(500)
        else:
            print(ret, fname)
    img = cv2.imread(images[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    print("ret", ret, "mtx", mtx, "dist", dist, "rvecs", rvecs, "tvecs", tvecs)


try:
    input_n = int(sys.argv[1])
except IndexError:
    input_n = 40


get_n_frame('frames', 'frames', input_n)
get_camera_caliburation()
