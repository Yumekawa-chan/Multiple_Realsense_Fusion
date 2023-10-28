import open3d as o3d
import numpy as np
import os


def get_file_paths():
    pcd_folder = os.path.join("data")
    date_folder = input("Input date folder name: ")
    filename = input("Input file name: ")
    pcd_r_path = os.path.join(pcd_folder, date_folder,
                              "pointclouds", f"{filename}_right.ply")
    pcd_l_path = os.path.join(pcd_folder, date_folder,
                              "pointclouds", f"{filename}_left.ply")

    return pcd_r_path, pcd_l_path, filename


def load_point_clouds(pcd_r_path, pcd_l_path):
    pcd_r = o3d.io.read_point_cloud(pcd_r_path)  # Reference Point Cloud
    pcd_l = o3d.io.read_point_cloud(pcd_l_path)  # Point Cloud to Transform
    return pcd_r, pcd_l


def get_transformation_matrix(R, T):
    # Defining rotation and translation matrices
    rotation = R
    # m -> cm, and change shape from (3,1) to (3,)
    translation = (T * (-0.01)).squeeze()
    # Creating transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    return transformation_matrix


def main():
    pcd_r_path, pcd_l_path, filename = get_file_paths()
    pcd_r, pcd_l = load_point_clouds(pcd_r_path, pcd_l_path)

    camera_matrix = np.load(os.path.join(
        "data", "matrix", "camera_matrix.npz"))
    R = camera_matrix["R"]
    T = camera_matrix["T"]

    # Uncomment below lines if you want to convert the point cloud to a single color.
    # pcd_r.paint_uniform_color([1, 0, 0])
    # pcd_l.paint_uniform_color([0, 1, 0])

    transformation_matrix = get_transformation_matrix(R, T)

    # Applying rotation and translation to pcd_l
    pcd_l.transform(transformation_matrix)

    # Merging the two point clouds
    merged_pcd = pcd_r + pcd_l

    # Displaying result (optional)
    o3d.visualization.draw_geometries([merged_pcd])

    # Saving result
    o3d.io.write_point_cloud(os.path.join(
        "data", "results", f"{filename}.ply"), merged_pcd)


if __name__ == "__main__":
    main()
