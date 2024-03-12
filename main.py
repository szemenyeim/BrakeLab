import numpy as np
from emergency_brake import EmergencyBrake
from pcp import PointCloudProcess
from camera_controll import CameraControll
from utils import initialize_with_config
import argparse
import pyrealsense2 as rs
import cv2


def visualize_cluster_centers(object_list: list, camera, img: np.ndarray) -> None:
    # List to store center points of objects and their corresponding colors
    center_point_list = []

    # Mapping of object status to colors: 0 - Green, 1 - Orange, 2 - Red
    color_mapping = {0: (0, 255, 0), 1: (0, 165, 255), 2: (0, 0, 255)}

    # Iterate through each object in the object list
    for object_item in object_list:
        # Project the center position of the object onto the image plane using camera intrinsic parameters
        center_point = rs.rs2_project_point_to_pixel(camera.intrinsic_pyrealsense, object_item.center_pos)

        # Get the color corresponding to the object's status from the color mapping
        status_color = color_mapping.get(object_item.status, (255, 255, 255))  # Default to white if status is not in mapping

        # Append the center point and its color to the list
        center_point_list.append((center_point, status_color))

    # Draw circles at the center points of objects on the image
    [cv2.circle(img, (int(round(center_point[0])), int(round(center_point[1]))), radius=5, color=color, thickness=-1)
     for center_point, color in center_point_list]



def setup_and_run(config:str, visualize: bool = False)->None:
    camera = initialize_with_config(CameraControll, config_file=config, type="camera")
    process = initialize_with_config(PointCloudProcess, config_file=config, type="point_cloud", camera=camera.camera)
    emergency=initialize_with_config(EmergencyBrake,config_file=config,type="emergency_brake")
    status=True

    while True:
        try:
            point_cloud, img = camera.get_next_frame(type="point_cloud")
        except EOFError as e:
            camera.logger.error(e)
            break
        objects = process(point_cloud)
        status=emergency(objects)

        if visualize:
            visualize_cluster_centers(objects, camera.camera, img)
            cv2.imshow('Objects Visualization', img)  # Display the frame with objects
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Press 'q' to exit
    # TODO if status False -> stop the car

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize with config file")
    parser.add_argument("--config", type=str, help="Path to the config file", required=False, default="./config.json")
    parser.add_argument("--visualize", type=bool, help="Show the cluster center points", required=False, default=True)
    args = parser.parse_args()
    setup_and_run(args.config, args.visualize)
