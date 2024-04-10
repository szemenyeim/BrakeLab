from utils import Object, Camera
import open3d as o3d
import numpy as np
import copy
import math
import time
import cv2 as cv
import logging
import colorlog
import pyrealsense2 as rs


class PointCloudProcess:
    """
        Class for processing point clouds.

    """
    point_cloud = None
    plane_model = None
    object_list = list()
    pre_downsample=3


    def __init__(self, camera: Camera, **kwargs) -> None:
        """
            Initialize PointCloudProcess object.

            Args:
                camera (Camera): Camera object.
                **kwargs: Additional keyword arguments.
        """
        self.camera = camera
        self.kmeans_n_cluster = kwargs.pop("kmeans_n_cluster", 10)
        self.ransac_distance_threshold = kwargs.pop("ransac_distance_threshold", 0.04)
        self.ransac_n = kwargs.pop("ransac_n", 3)
        self.ransac_num_iterations = kwargs.pop("num_iterations", 1000)
        self.show_first_process = kwargs.pop("show_first_process", True)
        self.radius_outlier_nb_points = kwargs.pop("radius_outlier_nb_points", 10)
        self.radius_outlier_radius = kwargs.pop("radius_outlier_radius", 0.3)
        self.max_distance_to_keep = kwargs.pop("max_distance_to_keep", 2.5)
        self.min_distance_to_keep = kwargs.pop("min_distance_to_keep", 0)
        self.first = True
        self.define_color_logger()

    def define_color_logger(self):
        # Create a logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Create a colorized formatter
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={},
            style='%'
        )

        # Create a console handler and set the formatter
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)

        # Add the console handler to the logger
        self.logger.addHandler(ch)

    def make_object_list_opencv(self, label: np.ndarray, center: np.ndarray) -> list:
        """
        Create a list of Object instances using OpenCV labels and centers.

        Args:
            label (numpy.ndarray): Array of labels.
            center (numpy.ndarray): Array of centers.

        Returns:
            list: List of Object instances.
        """
        # Convert label array to numpy array
        labels = np.array(label.tolist())
        # Create a list of Object instances based on labels and centers
        object_list = [Object(center_pos=center[x, :], num=np.count_nonzero(labels == x)) for x in
                       range(self.kmeans_n_cluster)]
        return object_list

    @staticmethod
    def remove_points(points: np.ndarray, criterium: float, max_or_min: bool) -> np.ndarray:
        """
        Remove points based on a criteria.

        Args:
            points (numpy.ndarray): Array of points.
            criterium (float): Criterion for removing points.
            max_or_min (bool): Flag indicating whether to remove points above or below the criterion.

        Returns:
            numpy.ndarray: Updated array of points after removal.
        """
        # Find indices of points to delete based on criteria
        delete = np.where(points[:, 2] > criterium) if max_or_min else np.where(points[:, 2] < criterium)
        # Generate indices of remaining points
        indexes = list(range(0, points[:, 2].size))
        # Delete points based on indices
        points_new = np.delete(indexes, delete)
        # Update points array
        points = points[points_new]
        return points

    def point_cloud_prefilter(self) -> None:
        """
            Apply pre-filtering to the point cloud.

            This method removes points based on maximum and minimum distance criteria.
            """
        # Create a deep copy of the point cloud points
        points = copy.deepcopy(np.asarray(self.point_cloud.points))

        # TODO: Remove points beyond the maximum distance

        # TODO: Remove points below the minimum distance

        # TODO: Downsample points by selecting every x point

        # Convert the filtered points to an Open3D point cloud
        vector = o3d.pybind.utility.Vector3dVector(points)
        point_cloud = o3d.geometry.PointCloud(vector)

        # Update the point cloud attribute with the pre-filtered point cloud
        self.point_cloud = copy.deepcopy(point_cloud)

    def calculate_plane_points(self) -> np.ndarray:
        """
        Calculate inlier points of the plane based on the current plane model.

        Returns:
            numpy.ndarray: Indices of inlier points representing the plane.
        """
        # Extract plane parameters (a, b and c are the normal)
        [a, b, c, d] = self.plane_model
        # Get point cloud points
        points = np.asarray(self.point_cloud.points)
        # TODO: Calculate length of the plane normal

        # TODO: Calculate distances from points to the plane

        # TODO: Create a mask for inlier points based on distance threshold

        # TODO: Extract indices of inlier points

        return inlier_points

    def remove_ground_points(self, inliers_points: np.ndarray) -> None:
        """
            Remove ground points from the point cloud.

            Args:
                inliers_points (numpy.ndarray): Indices of inliers representing ground points.

            """
        # Create a deep copy of the point cloud points
        points = copy.deepcopy(np.asarray(self.point_cloud.points))
        # TODO: Generate indices of all points

        # TODO: Delete inliers (ground points)

        # TODO: Update points array

        # Convert points to an Open3D point cloud
        vector = o3d.pybind.utility.Vector3dVector(points)
        point_cloud = o3d.geometry.PointCloud(vector)
        # Update the point cloud attribute
        self.point_cloud = copy.deepcopy(point_cloud)
        # Remove radius outliers
        self.point_cloud.remove_radius_outlier(nb_points=self.radius_outlier_nb_points,
                                               radius=self.radius_outlier_radius)

    def ransac(self) -> None:
        """
        Perform RANSAC plane segmentation on the point cloud and remove ground points.

        """
        start = time.time()
        # Visualize point cloud before pre-filtering if it's the first run and show_first_process is enabled
        if self.first and self.show_first_process:
            o3d.visualization.draw_geometries([self.point_cloud], window_name="Original point cloud",zoom=0.1,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[0, 0, 0],
                                  up=[-0.0694, -0.9768, 0.2024])

        # TODO: Apply pre-filtering to the point cloud


        # Visualize point cloud after pre-filtering if it's the first run and show_first_process is enabled
        if self.first and self.show_first_process:
            o3d.visualization.draw_geometries([self.point_cloud], window_name="After prefilter",zoom=0.1,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[0, 0, 0],
                                  up=[-0.0694, -0.9768, 0.2024])

        # Segment plane using RANSAC
        if self.plane_model is None:
            # TODO: If plane model is not yet initialized, perform RANSAC

            self.init_num_iniliers=len(inliers_points)
            self.plane_model = plane_model
        else:
            # TODO: Otherwise, calculate plane points using the existing model

            if len(inliers_points)<0.8*self.init_num_iniliers:
                self.pre_downsample+=1
                self.pre_downsample=7 if self.pre_downsample>7 else self.pre_downsample
                self.init_num_iniliers=len(inliers_points)

        # TODO: Remove ground points


        # Visualize point cloud after ground removal if it's the first run and show_first_process is enabled
        if self.first and self.show_first_process:
            o3d.visualization.draw_geometries([self.point_cloud], window_name="After RANSAC",zoom=0.1,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[0, 0, 0],
                                  up=[-0.0694, -0.9768, 0.2024])

        self.first = False
        end = time.time()
        self.logger.info(f"RANSAC running time: {end - start}")

    def kmeans_prefilter(self,downsample:int=3) -> None:
        """
        Apply pre-filtering to the point cloud using k-means clustering.
         Parameters:
        - downsample (int): The downsampling factor.

        """
        # Create a deep copy of the point cloud points
        points = copy.deepcopy(np.asarray(self.point_cloud.points))
        # Downsample points by selecting every x point
        points_new = points[::downsample]
        # Convert the downsampled points to an Open3D point cloud
        vector = o3d.pybind.utility.Vector3dVector(points_new)
        point_cloud = o3d.geometry.PointCloud(vector)
        # Update the point cloud attribute with the pre-filtered point cloud
        self.point_cloud = copy.deepcopy(point_cloud)

    def kmeans_opencv(self) -> None:
        """
        Apply k-means clustering using OpenCV.

        """
        # Start timing
        start = time.time()

        # TODO: Apply pre-filtering to the point cloud

        if len(self.point_cloud.points)>15:
            # TODO: Define termination criteria for k-means

            # TODO: Perform k-means clustering

            # Create object list from labels and centers
            self.object_list = self.make_object_list_opencv(label=label, center=center)
            # End timing
        end = time.time()
        # Log running time
        self.logger.info(f"K-means running time: {end - start}")


    def __call__(self, point_cloud):
        self.point_cloud = point_cloud
        self.logger.warning(f"Initial: {len(self.point_cloud.points)}")
        self.ransac()
        self.kmeans_opencv()
        return self.object_list
