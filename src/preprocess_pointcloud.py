import open3d as o3d
import os
import numpy as np
from tqdm import tqdm


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


def color_to_rgb(pcd):
    # Convert point cloud colors to numpy array
    colors = np.asarray(pcd.colors)
    # Swap the colors (assuming BGR to RGB)
    colors = colors[:, [2, 1, 0]]
    # Set the colors back to the point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def remove_outliers(pcd, nb_neighbors=50, std_ratio=2.0, radius=0.05, min_nb_points=10):
    for _ in tqdm(range(2), desc="Processing outliers"):
        pcd, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        pcd, ind = pcd.remove_radius_outlier(
            nb_points=min_nb_points, radius=radius)
    return pcd


def process_outliers(pcd_r, pcd_l):
    print("Processing the reference point cloud...")
    pcd_r = color_to_rgb(pcd_r)  # Convert colors to RGB
    pcd_r_processed = remove_outliers(pcd_r)
    print("Processing the point cloud to transform...")
    pcd_l = color_to_rgb(pcd_l)  # Convert colors to RGB
    pcd_l_processed = remove_outliers(pcd_l)
    return pcd_r_processed, pcd_l_processed


def save_processed_pointclouds(pcd_r_processed, pcd_l_processed, pcd_r_path, pcd_l_path):
    o3d.io.write_point_cloud(pcd_r_path, pcd_r_processed)
    o3d.io.write_point_cloud(pcd_l_path, pcd_l_processed)


def main():
    pcd_r_path, pcd_l_path, filename = get_file_paths()
    pcd_r, pcd_l = load_point_clouds(pcd_r_path, pcd_l_path)

    pcd_r_processed, pcd_l_processed = process_outliers(pcd_r, pcd_l)
    save_processed_pointclouds(
        pcd_r_processed, pcd_l_processed, pcd_r_path, pcd_l_path)

    # Visualizing result
    o3d.visualization.draw_geometries([pcd_r_processed, pcd_l_processed])


if __name__ == "__main__":
    main()
