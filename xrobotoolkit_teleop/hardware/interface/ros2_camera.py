import threading

import cv2
import numpy as np
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

from ...utils.image_utils import compress_image_to_jpg
from .base_camera import BaseCameraInterface

try:
    from cv_bridge import CvBridge
except Exception:
    CvBridge = None


CAMERA_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    durability=DurabilityPolicy.VOLATILE,
)


class Ros2CameraInterface(Node, BaseCameraInterface):
    """
    ROS2 camera interface for one or more image topics.
    """

    def __init__(
        self,
        node_name: str,
        camera_topics: dict,
        enable_depth: bool = True,
        width: int = None,
        height: int = None,
        enable_compression: bool = True,
        jpg_quality: int = 85,
    ):
        Node.__init__(self, node_name)
        BaseCameraInterface.__init__(
            self,
            enable_compression=enable_compression,
            jpg_quality=jpg_quality,
        )
        self.camera_topics = camera_topics
        self.enable_depth = enable_depth
        self.width = width
        self.height = height
        self.bridge = CvBridge() if CvBridge is not None else None
        self._warned_depth_bridge = False

        self.frames_dict = {}
        self.compressed_frames_dict = {}
        self.frames_lock = threading.Lock()
        self.subscribers = []

    def start(self):
        for name, topics in self.camera_topics.items():
            if "color" in topics:
                color_topic = topics["color"]
                msg_type = CompressedImage if color_topic.endswith("/compressed") else Image
                callback = self._color_compressed_callback if msg_type is CompressedImage else self._color_raw_callback
                self.subscribers.append(
                    self.create_subscription(
                        msg_type,
                        color_topic,
                        lambda msg, camera_name=name, cb=callback: cb(msg, camera_name),
                        CAMERA_QOS,
                    )
                )
            if self.enable_depth and "depth" in topics:
                depth_topic = topics["depth"]
                depth_type = CompressedImage if depth_topic.endswith("/compressed") else Image
                depth_cb = self._depth_compressed_callback if depth_type is CompressedImage else self._depth_raw_callback
                self.subscribers.append(
                    self.create_subscription(
                        depth_type,
                        depth_topic,
                        lambda msg, camera_name=name, cb=depth_cb: cb(msg, camera_name),
                        CAMERA_QOS,
                    )
                )

    def stop(self):
        for sub in self.subscribers:
            self.destroy_subscription(sub)
        self.subscribers = []

    def _resize_image(self, image):
        if self.width is not None and self.height is not None and image is not None:
            return cv2.resize(image, (self.width, self.height))
        return image

    def _ensure_camera_entry(self, camera_name: str):
        if camera_name not in self.frames_dict:
            self.frames_dict[camera_name] = {}
            self.compressed_frames_dict[camera_name] = {}

    def _color_compressed_callback(self, msg: CompressedImage, camera_name: str):
        np_arr = np.frombuffer(msg.data, np.uint8)
        color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if color_image is None:
            return
        color_image = self._resize_image(color_image)

        with self.frames_lock:
            self._ensure_camera_entry(camera_name)
            self.frames_dict[camera_name]["color"] = color_image
            if self.enable_compression:
                self.compressed_frames_dict[camera_name]["color"] = bytes(msg.data)

    def _color_raw_callback(self, msg: Image, camera_name: str):
        if self.bridge is None:
            return
        color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        color_image = self._resize_image(color_image)

        with self.frames_lock:
            self._ensure_camera_entry(camera_name)
            self.frames_dict[camera_name]["color"] = color_image
            if self.enable_compression:
                self.compressed_frames_dict[camera_name]["color"] = compress_image_to_jpg(
                    color_image,
                    self.jpg_quality,
                )

    def _depth_compressed_callback(self, msg: CompressedImage, camera_name: str):
        np_arr = np.frombuffer(msg.data, np.uint8)
        depth_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            return
        depth_image = self._resize_image(depth_image)

        with self.frames_lock:
            self._ensure_camera_entry(camera_name)
            self.frames_dict[camera_name]["depth"] = depth_image
            if self.enable_compression:
                self.compressed_frames_dict[camera_name]["depth"] = bytes(msg.data)

    def _depth_raw_callback(self, msg: Image, camera_name: str):
        if self.bridge is None:
            if not self._warned_depth_bridge:
                self.get_logger().warning(
                    "cv_bridge is unavailable; skipping raw depth frames."
                )
                self._warned_depth_bridge = True
            return

        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth_image = self._resize_image(depth_image)

        with self.frames_lock:
            self._ensure_camera_entry(camera_name)
            self.frames_dict[camera_name]["depth"] = depth_image
            if self.enable_compression:
                self.compressed_frames_dict[camera_name]["depth"] = compress_image_to_jpg(
                    depth_image,
                    self.jpg_quality,
                )

    def update_frames(self):
        pass

    def get_frames(self):
        with self.frames_lock:
            frames_dict = {}
            for camera_name, frame_data in self.frames_dict.items():
                color_frame = frame_data.get("color")
                depth_frame = frame_data.get("depth")
                frames_dict[camera_name] = {
                    "color": color_frame.copy() if color_frame is not None else None,
                    "depth": depth_frame.copy() if self.enable_depth and depth_frame is not None else None,
                }
            return frames_dict

    def get_compressed_frames(self):
        with self.frames_lock:
            compressed_dict = {}
            for camera_name, frame_data in self.compressed_frames_dict.items():
                color_bytes = frame_data.get("color")
                depth_bytes = frame_data.get("depth")
                compressed_dict[camera_name] = {
                    "color": color_bytes[:] if color_bytes is not None else None,
                    "depth": depth_bytes[:] if self.enable_depth and depth_bytes is not None else None,
                }
            return compressed_dict

    def get_frame(self, camera_name: str):
        with self.frames_lock:
            frame_data = self.frames_dict.get(camera_name, {})
            color_frame = frame_data.get("color")
            depth_frame = frame_data.get("depth")
            return {
                "color": color_frame.copy() if color_frame is not None else None,
                "depth": depth_frame.copy() if self.enable_depth and depth_frame is not None else None,
            }
