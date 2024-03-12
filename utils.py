import open3d as o3d
import numpy as np
import json
import typing
import pyrealsense2 as rs


def read_json_config(config_file):
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Config file '{config_file}' not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Unable to parse JSON in '{config_file}'.")
        return {}


def initialize_with_config(class_obj: typing.Callable, config_file: str, type: str = None, **kwargs):
    config = read_json_config(config_file)
    if type is not None:
        config = config[type]
    config.update(kwargs)
    return class_obj(**config)


class Camera:
    w = None
    h = None
    fx = None
    fy = None
    ppx = None
    ppy = None
    o3dstruct = None
    converter = None

    def __init__(self, w: int, h: int, fx: float, fy: float, ppx: float, ppy: float, py_intri=None):
        self.fx = fx
        self.fy = fy
        self.h = h
        self.w = w
        self.ppx = ppx
        self.ppy = ppy
        self.intrinsic = np.asarray([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
        self.o3dstruct = o3d.camera.PinholeCameraIntrinsic(self.w, self.h, self.fx, self.fy, self.ppx, self.ppy)
        if py_intri is not None:
            self.intrinsic_pyrealsense = py_intri
        else:
            intrinsics = rs.intrinsics()
            # Populate the intrinsic parameters object with values from the intrinsic matrix
            intrinsics.width = w
            intrinsics.height = h
            intrinsics.fx = fx
            intrinsics.fy = fy
            intrinsics.ppx = ppx
            intrinsics.ppy = ppy

            # Assign the intrinsic parameters to the intrinsics object
            intrinsics.coeffs = [0, 0, 0, 0, 0]  # Assuming no distortion
            self.intrinsic_pyrealsense = intrinsics


class Object:
    life = None             # Life of the object
    center_pos = None       # Position of the object's center
    num_points = None       # Number of points in the object
    max_life = 5            # Maximum life value
    min_life = -3           # Minimum life value
    status = 0              # Status of the object: 0 - good, 1 - Warning, 2 - Stop

    def __init__(self, center_pos, num):
        # Initialize the Object with its center position and number of points.
        self.life = 0
        self.center_pos = center_pos
        self.num_points = num

    def ageing(self):
        # Decrease the life of the object by 1.
        self.life = self.life - 1

    def check_distance(self, min_x: float, min_y: float, min_z: float, security_margin: float) -> bool:
        # Check if the object's number of points is greater than 50 and if it's within the distance thresholds.
        if self.num_points > 50:
            if abs(self.center_pos[0]) < min_x or abs(self.center_pos[1]) < min_y or abs(self.center_pos[2]) < min_z:
                # If the object is too close, set status to Stop (2) and return False.
                self.status = 2
                return False
            if abs(self.center_pos[0]) < (min_x + security_margin) or abs(self.center_pos[1]) < (min_y + security_margin) or \
                    abs(self.center_pos[2]) < (min_z + security_margin):
                # If the object is within the security margin, set status to Warning (1) and return True.
                self.status = 1
                return True
        # If the object is not too close or doesn't have enough points, set status to good (0) and return True.
        self.status = 0
        return True



class Point:
    x = None
    y = None
    z = None

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
