# Multiple_Realsense_Fusion

Combine multiple point clouds from multiple Realsense devices through external calibration. This source code is based on the [Proceedings of the Seventh IEEE International Conference on Computer Vision](https://ieeexplore.ieee.org/document/791183).

## Procedure

1. **External Calibration**:

   - Place two sensors at arbitrary positions.

   <img src="assets\images\sensors.jpg" width = 500 title="Example of sensor placement">

   - Execute `realsense_stereo_calib.py`.
   - Utilize a checkerboard for calibration. A single calibration action is executed with a right-click.
   - It is recommended to capture between 30 to 40 images.

    <img src="assets\images\calibration.jpg" width = 500 title="Calibration using a checkerboard">
     
   - Enter the checkerboard square size (in cm) into the source code (default is 4.75cm).

2. **Point Cloud Capture**:

   - Use `realsense_data_capture` to capture the point clouds you wish to merge.
   - Press the 's' key to capture.

3. **Outlier Removal**:

   - Execute `preprocess_pointcloud.py` to remove outliers from the point cloud.

4. **Point Cloud Merging**:
   - Execute `pointcloudMerger.py` to perform the merging of point clouds.

## Example of result

It is recommended to adjust minor misalignments using ICP (Iterative Closest Point) matching.
<img src="assets\images\result_sample.png" width = 500 title="Example of result">
