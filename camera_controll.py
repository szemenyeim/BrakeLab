import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import cv2
from utils import Camera
import os
import glob
import logging
import colorlog
import json


class CameraControll:
    index=0
    """
    Class for controlling the camera.

    Attributes:
        pipeline: Pipeline for processing frames.
        online (bool): Flag indicating if camera is online (default is True).
        path (str): Path to offline images (default is None).
        start_end_index (tuple): Tuple containing indices for start and end frames (default is (0, -1)).
        fps (int): Frames per second (default is 30).
        image_size (tuple): Tuple containing image width and height (default is (480, 848)).
    """

    def __init__(self, **kwargs):
        """
        Initialize CameraControll object.

        Args:
            **kwargs: Additional keyword arguments.
        """
        # Initialize attributes
        self.pipeline = None
        self.online = kwargs.pop("online", True)
        self.path = kwargs.pop("path", None)
        self.start_end_index = kwargs.pop("start_end_index", (0, -1))
        self.fps = kwargs.pop("fps", 30)
        self.image_size = kwargs.pop("image_size", (480, 848))

        # Define color logger
        self.define_color_logger()
        if self.online == False and self.path is None:
            raise ValueError("Path input required for offline use.")
        # Start pipeline if online, otherwise setup for offline usage
        if self.online:
            self.start_pipeline()
        else:
            self.setup_offline_usage()

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

    def setup_pipeline(self) -> None:
        """
        Set up the online measurement pipeline by configuring the camera to capture
        color and depth frames at a specified frames per second (fps) and image size.
        This function initializes the RealSense pipeline, configures the streams, and
        starts capturing frames.

        """
        # Log the start of the online measurement setup
        self.logger.info(
            f"Start online measurement, setup camera to color and depth frames with {self.fps} fps and {self.image_size} image size")

        # Configure RealSense pipeline
        config = rs.config()
        self.pipeline = rs.pipeline()

        # Enable depth and color streams with specified parameters
        config.enable_stream(rs.stream.depth, self.image_size[1], self.image_size[0], rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.image_size[1], self.image_size[0], rs.format.bgr8, self.fps)

        # Start capturing frames
        profile = self.pipeline.start(config)

        # Align color and depth frames
        self.aligned_stream = rs.align(rs.stream.color)

        # Get camera parameters
        streams = profile.get_streams()
        info = streams[0].as_video_stream_profile().get_intrinsics()

        # Initialize camera object with intrinsic parameters
        self.camera = Camera(info.width, info.height, info.fx, info.fy, info.ppx, info.ppy, info)

    def setup_offline_usage(self) -> None:
        """
        Sets up the offline usage by initializing necessary parameters and loading
        required files.
        :return:
        """
        # Log the start of the offline measurement process
        self.logger.info(
            f"Start offline measurement, read files from {self.path}/images")

        # Get paths for depth and RGB images
        self.depth_images = glob.glob(f"{self.path}/images/depth/*.*")
        self.rgb_images = glob.glob(f"{self.path}/images/rgb/*.*")

        # Check if there are images available
        if len(self.depth_images) < 1 or len(self.rgb_images) < 1:
            raise ValueError(f"Can not find the images under {self.path}/images")
        # Sort images by index
        self.depth_images.sort(key=lambda x: int(x.split('/')[-1].split("_")[-1].split('.')[0]))
        self.rgb_images.sort(key=lambda x: int(x.split('/')[-1].split("_")[-1].split('.')[0]))

        # Load camera intrinsic matrix
        with open(f"{self.path}/camera_intrinsic_matrix.json", "r") as file:
            camera_intrinsics = np.asarray(json.load(file))

        # Check if camera matrix is loaded successfully
        if camera_intrinsics is None:
            raise ValueError("Camera matrix cannot be None")

        # Create camera object with loaded intrinsic parameters
        self.camera = Camera(self.image_size[1], self.image_size[0], camera_intrinsics[0, 0], camera_intrinsics[1, 1],
                             camera_intrinsics[0, 2], camera_intrinsics[1, 2])

    def get_image(self):
        """
        Return pairs of RGB and depth images.
        """
        # Iterate through the indices of the image lists
        end_index = len(self.depth_images)-1 if self.start_end_index[1] == -1 else self.start_end_index[1]
        index=self.start_end_index[0]+self.index
        if index>end_index:
            self.logger.warning("Run out of images.")
            return None, None

        # Read the RGB and depth images
        img_rgb = cv2.imread(self.rgb_images[index])
        img_depth = cv2.imread(self.depth_images[index], cv2.IMREAD_UNCHANGED)

        # Check if either of the images is None and raise an error if any of the images is missing
        if img_depth is None or img_rgb is None:
            raise ValueError(f"Error while opening {self.rgb_images[index]}, {self.depth_images[index]} images")
        # Update index
        self.index+=1
        # Return the RGB and depth images as a tuple
        return img_rgb, img_depth

    def get_next_frame(self, type: str, **kwargs):
        """
        Retrieve the next frame based on the specified type.

        Args:
            type (str): Type of frame to retrieve, either "depth" or "point_cloud".
            **kwargs: Additional keyword arguments.
                every_k_points (int): Number of points to downsample in point cloud (default is 6).

        Returns:
            tuple: Tuple containing frame depth and RGB image if type is "depth",
                                               otherwise returns a downsampled point cloud and RGB image.

      """
        # Extract every_k_points from kwargs, default to 6 if not provided
        every_k_points = kwargs.pop("every_k_points", 6)

        if self.online:
            # Get frames with 5000 ms timeout
            frames = self.pipeline.wait_for_frames(5000)
            frames = self.aligned_stream.process(frames)
            frame_depth = np.asanyarray(frames.get_depth_frame().get_data())
            frame_rgb = np.asarray(frames.get_color_frame().get_data())
        else:
            # If not online, retrieve frames from the image generator
             frame_rgb,frame_depth = self.get_image()

        if frame_rgb is None or frame_depth is None:
            raise EOFError("At least one of the frames is None. End of file reached or camera died.")

        frame_depth=cv2.resize(frame_depth,(self.image_size[1],self.image_size[0]),interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.resize(frame_rgb, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
        if type == "depth":
            return frame_depth, frame_rgb
        elif type == "point_cloud":
            # Convert depth image to Open3D image
            o3d_image = o3d.geometry.Image(frame_depth)
            # Create point cloud from depth image
            source_pcd = o3d.geometry.PointCloud().create_from_depth_image(o3d_image, self.camera.o3dstruct)
            # Downsample point cloud
            point_cloud = source_pcd.uniform_down_sample(every_k_points=every_k_points)
            return point_cloud, frame_rgb
        else:
            # Raise error if type is not supported
            raise ValueError("Only depth and point_cloud types are supported by the get_next_frame function.")
