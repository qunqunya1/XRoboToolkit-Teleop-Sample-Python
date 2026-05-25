import threading
import sys

import cv2
import numpy as np
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

from ...utils.image_utils import compress_image_to_jpg
from .base_camera import BaseCameraInterface


CAMERA_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    durability=DurabilityPolicy.VOLATILE,
)


def _compress_depth_to_png(depth_image):
    ok, encoded_img = cv2.imencode(".png", depth_image)
    if not ok:
        return None
    return encoded_img.tobytes()


def _encoding_to_dtype_and_channels(encoding: str):
    normalized = encoding.lower()
    if normalized in {"bgr8", "rgb8"}:
        return np.uint8, 3
    if normalized in {"mono8", "8uc1"}:
        return np.uint8, 1
    if normalized in {"mono16", "16uc1"}:
        return np.uint16, 1
    if normalized == "32fc1":
        return np.float32, 1
    return None, None


def _is_compressed_topic(topic: str) -> bool:
    return topic.endswith("/compressed") or topic.endswith("_compressed")


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
        decode_compressed_for_display: bool = False,
        raw_passthrough_for_logging: bool = False,
    ):
        Node.__init__(self, node_name)
        BaseCameraInterface.__init__(
            self,
            enable_compression=enable_compression,
            jpg_quality=jpg_quality,
        )
        self.camera_topics = camera_topics
        self.enable_depth = enable_depth
        self.width = int(width) if width and int(width) > 0 else None
        self.height = int(height) if height and int(height) > 0 else None
        self.decode_compressed_for_display = bool(decode_compressed_for_display)
        self.raw_passthrough_for_logging = bool(raw_passthrough_for_logging)
        self._warned_raw_encodings = set()

        self.frames_dict = {}
        self.compressed_frames_dict = {}
        self.frames_lock = threading.Lock()
        self.subscribers = []

    def start(self):
        for name, topics in self.camera_topics.items():
            if "color" in topics:
                color_topic = topics["color"]
                msg_type = CompressedImage if _is_compressed_topic(color_topic) else Image
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
                depth_type = CompressedImage if _is_compressed_topic(depth_topic) else Image
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

    def _image_msg_to_numpy(self, msg: Image):
        dtype, channels = _encoding_to_dtype_and_channels(msg.encoding)
        if dtype is None:
            if msg.encoding not in self._warned_raw_encodings:
                self.get_logger().warning(
                    f"Unsupported raw image encoding '{msg.encoding}'; skipping frame."
                )
                self._warned_raw_encodings.add(msg.encoding)
            return None

        itemsize = np.dtype(dtype).itemsize
        width = int(msg.width)
        height = int(msg.height)
        step = int(msg.step) if msg.step else width * channels * itemsize
        row_items = step // itemsize
        expected_items = height * row_items
        image = np.frombuffer(msg.data, dtype=dtype, count=expected_items)
        if bool(getattr(msg, "is_bigendian", False)) != (sys.byteorder == "big"):
            image = image.byteswap()
        image = image.reshape((height, row_items))
        image = image[:, : width * channels]
        if channels > 1:
            image = image.reshape((height, width, channels))
        else:
            image = image.reshape((height, width))

        if msg.encoding.lower() == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def _color_compressed_callback(self, msg: CompressedImage, camera_name: str):
        if not self.decode_compressed_for_display and self.width is None and self.height is None:
            with self.frames_lock:
                self._ensure_camera_entry(camera_name)
                if self.enable_compression:
                    self.compressed_frames_dict[camera_name]["color"] = bytes(msg.data)
            return

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
        if (
            self.raw_passthrough_for_logging
            and not self.enable_compression
            and self.width is None
            and self.height is None
        ):
            with self.frames_lock:
                self._ensure_camera_entry(camera_name)
                self.frames_dict[camera_name]["color"] = {
                    "raw": bytes(msg.data),
                    "encoding": str(msg.encoding),
                    "width": int(msg.width),
                    "height": int(msg.height),
                    "step": int(msg.step),
                    "is_bigendian": bool(getattr(msg, "is_bigendian", False)),
                }
            return

        color_image = self._image_msg_to_numpy(msg)
        if color_image is None:
            return
        if color_image.ndim == 2:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_GRAY2BGR)
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
        if not self.decode_compressed_for_display and self.width is None and self.height is None:
            with self.frames_lock:
                self._ensure_camera_entry(camera_name)
                if self.enable_compression:
                    self.compressed_frames_dict[camera_name]["depth"] = bytes(msg.data)
            return

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
        if (
            self.raw_passthrough_for_logging
            and not self.enable_compression
            and self.width is None
            and self.height is None
        ):
            with self.frames_lock:
                self._ensure_camera_entry(camera_name)
                self.frames_dict[camera_name]["depth"] = {
                    "raw": bytes(msg.data),
                    "encoding": str(msg.encoding),
                    "width": int(msg.width),
                    "height": int(msg.height),
                    "step": int(msg.step),
                    "is_bigendian": bool(getattr(msg, "is_bigendian", False)),
                }
            return

        depth_image = self._image_msg_to_numpy(msg)
        if depth_image is None:
            return
        depth_image = self._resize_image(depth_image)

        with self.frames_lock:
            self._ensure_camera_entry(camera_name)
            self.frames_dict[camera_name]["depth"] = depth_image
            if self.enable_compression:
                self.compressed_frames_dict[camera_name]["depth"] = _compress_depth_to_png(depth_image)

    def update_frames(self):
        pass

    def get_frames(self):
        with self.frames_lock:
            frames_dict = {}
            for camera_name, frame_data in self.frames_dict.items():
                color_frame = frame_data.get("color")
                depth_frame = frame_data.get("depth")
                frames_dict[camera_name] = {
                    "color": self._copy_frame_value(color_frame),
                    "depth": self._copy_frame_value(depth_frame) if self.enable_depth and depth_frame is not None else None,
                }
            return frames_dict

    @staticmethod
    def _copy_frame_value(value):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, dict) and "raw" in value:
            return dict(value)
        return value

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
                "color": self._copy_frame_value(color_frame),
                "depth": self._copy_frame_value(depth_frame) if self.enable_depth and depth_frame is not None else None,
            }
