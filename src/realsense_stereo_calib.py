import numpy as np
import cv2
import pyrealsense2 as rs
from time import sleep

# Flag to detect right mouse click
right_click = False

# Mouse callback function


def mouse_callback(event, x, y, flags, param):
    global right_click
    if event == cv2.EVENT_RBUTTONDOWN:
        right_click = True


# Checkerboard configuration
CHECKERBOARD = (7, 10)
square_size = 4.75  # Size of each square on the checkerboard

# Preparing 3D points
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                          0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D points
imgpoints1 = []  # Camera 1's 2D points
imgpoints2 = []  # Camera 2's 2D points

# RealSense configuration
pipeline1 = rs.pipeline()
pipeline2 = rs.pipeline()
config1 = rs.config()
config2 = rs.config()
config1.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
config2.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
ctx = rs.context()
devices = ctx.query_devices()
serial_number_right = devices[0].get_info(rs.camera_info.serial_number)
serial_number_left = devices[1].get_info(rs.camera_info.serial_number)

config1.enable_device(serial_number_right)  # Device ID for Camera 1
config2.enable_device(serial_number_left)  # Device ID for Camera 2
profile1 = pipeline1.start(config1)
profile2 = pipeline2.start(config2)

# Obtaining internal parameters from RealSense
intrinsics1 = profile1.get_stream(
    rs.stream.color).as_video_stream_profile().get_intrinsics()
intrinsics2 = profile2.get_stream(
    rs.stream.color).as_video_stream_profile().get_intrinsics()

mtx1 = np.array([[intrinsics1.fx, 0, intrinsics1.ppx],
                 [0, intrinsics1.fy, intrinsics1.ppy],
                 [0, 0, 1]])

mtx2 = np.array([[intrinsics2.fx, 0, intrinsics2.ppx],
                 [0, intrinsics2.fy, intrinsics2.ppy],
                 [0, 0, 1]])

dist1 = np.array(intrinsics1.coeffs)
dist2 = np.array(intrinsics2.coeffs)

cv2.namedWindow('Right Camera')
cv2.namedWindow('Left Camera')
cv2.setMouseCallback('Right Camera', mouse_callback)
cv2.setMouseCallback('Left Camera', mouse_callback)

try:
    while True:
        frames1 = pipeline1.wait_for_frames()
        frames2 = pipeline2.wait_for_frames()

        color_frame1 = frames1.get_color_frame()
        color_frame2 = frames2.get_color_frame()

        color_image1 = np.asanyarray(color_frame1.get_data())
        color_image2 = np.asanyarray(color_frame2.get_data())

        color_image1 = cv2.cvtColor(color_image1, cv2.COLOR_RGB2BGR)
        color_image2 = cv2.cvtColor(color_image2, cv2.COLOR_RGB2BGR)

        gray1 = cv2.cvtColor(color_image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(color_image2, cv2.COLOR_BGR2GRAY)

        if right_click:  # Execute only if right mouse click is detected
            ret1, corners1 = cv2.findChessboardCorners(
                gray1, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            ret2, corners2 = cv2.findChessboardCorners(
                gray2, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

            if ret1 and ret2:
                objpoints.append(objp)
                imgpoints1.append(corners1)
                imgpoints2.append(corners2)

                # Drawing checkerboard corners
                cv2.drawChessboardCorners(
                    color_image1, CHECKERBOARD, corners1, ret1)
                cv2.drawChessboardCorners(
                    color_image2, CHECKERBOARD, corners2, ret2)
                sleep(0.5)

            right_click = False  # Resetting the flag

        cv2.imshow('Right Camera', color_image1)
        cv2.imshow('Left Camera', color_image2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline1.stop()
    pipeline2.stop()

print("Have a cup of coffee and wait for this long process!")
_, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
    objpoints, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2, gray1.shape[::-1])

print("Rotation Matrix:\n", R)
print("Translation Vector:\n", T)

np.savez_compressed(f'../data/matrix/camera_matrix', R=R, T=T)
