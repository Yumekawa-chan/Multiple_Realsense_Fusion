import cv2
import numpy as np
import pyrealsense2 as rs
from datetime import datetime
import os
import time


class RealSenseDevice:
    def __init__(self, serial_number):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial_number)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(
            rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pc = rs.pointcloud()
        self.pipeline.start(self.config)
        time.sleep(1)  # Add a short delay for stable streaming

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        return depth_frame, color_frame

    def get_pointcloud(self, depth_frame, color_frame):
        self.pc.map_to(color_frame)
        points = self.pc.calculate(depth_frame)
        return points

    def stop(self):
        self.pipeline.stop()


def save_data(depth_frame, color_frame, points, folder, suffix):
    timestamp = datetime.now().strftime("%H%M")
    cv2.imwrite(os.path.join(folder, 'depths', f'{timestamp}_{suffix}.png'),
                np.asanyarray(depth_frame.get_data()))
    cv2.imwrite(os.path.join(folder, 'colors', f'{timestamp}_{suffix}.png'),
                np.asanyarray(color_frame.get_data()))
    points.export_to_ply(os.path.join(
        folder, 'pointclouds', f'{timestamp}_{suffix}.ply'), color_frame)


def main():
    # Get the list of connected RealSense devices
    ctx = rs.context()
    devices = ctx.query_devices()

    # Initialize RealSense devices
    rs_device_right = RealSenseDevice(
        devices[0].get_info(rs.camera_info.serial_number))
    rs_device_left = RealSenseDevice(
        devices[1].get_info(rs.camera_info.serial_number))

    # Create directories for saving the data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data',
                            datetime.now().strftime("%Y%m%d"))
    os.makedirs(os.path.join(data_dir, 'pointclouds'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'depths'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'colors'), exist_ok=True)

    try:
        while True:
            # Get the frames from both devices
            depth_frame_right, color_frame_right = rs_device_right.get_frames()
            depth_frame_left, color_frame_left = rs_device_left.get_frames()

            # Generate pointclouds
            points_right = rs_device_right.get_pointcloud(
                depth_frame_right, color_frame_right)
            points_left = rs_device_left.get_pointcloud(
                depth_frame_left, color_frame_left)

            # Show the depth and color images
            cv2.imshow('Depth Image Right', np.asanyarray(
                depth_frame_right.get_data()))
            cv2.imshow('Color Image Right', np.asanyarray(
                color_frame_right.get_data()))
            cv2.imshow('Depth Image Left', np.asanyarray(
                depth_frame_left.get_data()))
            cv2.imshow('Color Image Left', np.asanyarray(
                color_frame_left.get_data()))

            key = cv2.waitKey(1)

            # Save the depth and color images when 's' is pressed
            if key & 0xFF == ord('s'):
                save_data(depth_frame_right, color_frame_right,
                          points_right, data_dir, 'right')
                save_data(depth_frame_left, color_frame_left,
                          points_left, data_dir, 'left')
                print("Saved!")

            # Exit when 'q' is pressed
            if key & 0xFF == ord('q'):
                break

    finally:
        # Stop the streams
        rs_device_right.stop()
        rs_device_left.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
