#!/usr/bin/env python3
"""
Receive and preview images published by realsense_camera.py.

Default settings match realsense_camera.py:
  rgb_transport=compressed -> rgb/image_compressed
  depth_transport=raw      -> depth/image_raw
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image


CAMERA_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


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


def image_msg_to_numpy(msg: Image) -> Optional[np.ndarray]:
    dtype, channels = _encoding_to_dtype_and_channels(msg.encoding)
    if dtype is None:
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


def compressed_msg_to_numpy(msg: CompressedImage, unchanged: bool = False) -> Optional[np.ndarray]:
    data = np.frombuffer(msg.data, np.uint8)
    flag = cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR
    return cv2.imdecode(data, flag)


def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    if depth.dtype == np.float32:
        finite = np.isfinite(depth)
        if not finite.any():
            scaled = np.zeros(depth.shape, dtype=np.uint8)
        else:
            max_value = float(np.nanpercentile(depth[finite], 95))
            alpha = 255.0 / max(1.0e-6, max_value)
            scaled = cv2.convertScaleAbs(depth, alpha=alpha)
    elif depth.dtype == np.uint16:
        nonzero = depth[depth > 0]
        max_value = int(np.percentile(nonzero, 95)) if nonzero.size else 1
        scaled = cv2.convertScaleAbs(depth, alpha=255.0 / max(1, max_value))
    else:
        scaled = cv2.convertScaleAbs(depth)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)


def put_label(image: np.ndarray, label: str) -> np.ndarray:
    out = image.copy()
    cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return out


class RealSenseImageReceiver(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("realsense_image_receiver")
        self.args = args
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_rgb_stamp = 0.0
        self.latest_depth_stamp = 0.0
        self.rgb_count = 0
        self.depth_count = 0
        self.last_status_time = time.time()

        self.save_dir = Path(args.save_dir).expanduser() if args.save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.video_writer = None
        if args.video_out:
            fourcc = cv2.VideoWriter_fourcc(*args.video_codec)
            self.video_writer = cv2.VideoWriter(args.video_out, fourcc, float(args.video_fps), (args.video_width, args.video_height))
            if not self.video_writer.isOpened():
                self.get_logger().error(f"Cannot open video writer: {args.video_out}")
                self.video_writer.release()
                self.video_writer = None

        if args.rgb_transport != "none":
            rgb_msg_type = CompressedImage if args.rgb_transport == "compressed" else Image
            rgb_cb = self._rgb_compressed_cb if args.rgb_transport == "compressed" else self._rgb_raw_cb
            self.create_subscription(rgb_msg_type, args.rgb_topic, rgb_cb, CAMERA_QOS)
            self.get_logger().info(f"Subscribing RGB {args.rgb_transport}: {args.rgb_topic}")

        if args.depth_transport != "none":
            depth_msg_type = CompressedImage if args.depth_transport == "compressed" else Image
            depth_cb = self._depth_compressed_cb if args.depth_transport == "compressed" else self._depth_raw_cb
            self.create_subscription(depth_msg_type, args.depth_topic, depth_cb, CAMERA_QOS)
            self.get_logger().info(f"Subscribing depth {args.depth_transport}: {args.depth_topic}")

        self.create_timer(1.0 / max(1.0, float(args.display_fps)), self._display_timer_cb)
        self.create_timer(1.0, self._status_timer_cb)

    def _rgb_raw_cb(self, msg: Image):
        image = image_msg_to_numpy(msg)
        if image is None:
            self.get_logger().warn(f"Unsupported RGB encoding: {msg.encoding}")
            return
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.latest_rgb = image
        self.latest_rgb_stamp = time.time()
        self.rgb_count += 1
        self._save_frame("rgb", image, self.rgb_count)

    def _rgb_compressed_cb(self, msg: CompressedImage):
        image = compressed_msg_to_numpy(msg)
        if image is None:
            self.get_logger().warn("Failed to decode compressed RGB frame")
            return
        self.latest_rgb = image
        self.latest_rgb_stamp = time.time()
        self.rgb_count += 1
        self._save_frame("rgb", image, self.rgb_count)

    def _depth_raw_cb(self, msg: Image):
        image = image_msg_to_numpy(msg)
        if image is None:
            self.get_logger().warn(f"Unsupported depth encoding: {msg.encoding}")
            return
        self.latest_depth = image
        self.latest_depth_stamp = time.time()
        self.depth_count += 1
        self._save_frame("depth", image, self.depth_count)

    def _depth_compressed_cb(self, msg: CompressedImage):
        image = compressed_msg_to_numpy(msg, unchanged=True)
        if image is None:
            self.get_logger().warn("Failed to decode compressed depth frame")
            return
        self.latest_depth = image
        self.latest_depth_stamp = time.time()
        self.depth_count += 1
        self._save_frame("depth", image, self.depth_count)

    def _save_frame(self, prefix: str, image: np.ndarray, index: int):
        if self.save_dir is None or self.args.save_every <= 0:
            return
        if index % self.args.save_every != 0:
            return
        suffix = "png" if prefix == "depth" else "jpg"
        cv2.imwrite(str(self.save_dir / f"{prefix}_{index:06d}.{suffix}"), image)

    def _display_timer_cb(self):
        if self.args.no_display:
            return

        tiles = []
        if self.latest_rgb is not None:
            tiles.append(put_label(self.latest_rgb, f"rgb #{self.rgb_count}"))
        if self.latest_depth is not None:
            tiles.append(put_label(depth_to_colormap(self.latest_depth), f"depth #{self.depth_count}"))
        if not tiles:
            return

        height = max(tile.shape[0] for tile in tiles)
        normalized = []
        for tile in tiles:
            if tile.shape[0] != height:
                scale = height / tile.shape[0]
                tile = cv2.resize(tile, (int(tile.shape[1] * scale), height))
            normalized.append(tile)
        canvas = np.hstack(normalized)

        if self.video_writer is not None:
            frame = cv2.resize(canvas, (self.args.video_width, self.args.video_height))
            self.video_writer.write(frame)

        cv2.imshow("realsense_receiver", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            rclpy.shutdown()
        elif key == ord("s"):
            snapshot_dir = self.save_dir or Path(".")
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(snapshot_dir / f"snapshot_{int(time.time())}.jpg"), canvas)

    def _status_timer_cb(self):
        now = time.time()
        rgb_gap = now - self.latest_rgb_stamp if self.latest_rgb_stamp else -1.0
        depth_gap = now - self.latest_depth_stamp if self.latest_depth_stamp else -1.0
        self.get_logger().info(
            f"frames: rgb={self.rgb_count}, depth={self.depth_count}, "
            f"rgb_gap={rgb_gap:.2f}s, depth_gap={depth_gap:.2f}s"
        )

    def destroy_node(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if not self.args.no_display:
            cv2.destroyAllWindows()
        return super().destroy_node()


def build_h265_udp_pipeline(args: argparse.Namespace, for_opencv: bool = True) -> str:
    if for_opencv:
        sink = "videoconvert ! video/x-raw,format=BGR ! appsink drop=true sync=false max-buffers=1"
    elif args.h265_sink == "fps":
        sink = "videoconvert ! fpsdisplaysink video-sink=autovideosink text-overlay=true sync=false"
    elif args.h265_sink == "fake":
        sink = "fakesink sync=false"
    else:
        sink = "videoconvert ! autovideosink sync=false"

    decode = "" if args.h265_sink == "fake" else "! avdec_h265 "
    return (
        f"udpsrc port={args.h265_port} "
        "caps=\"application/x-rtp,media=(string)video,encoding-name=(string)H265,"
        f"payload=(int){args.h265_payload}\" "
        f"! rtpjitterbuffer latency={args.h265_latency_ms} drop-on-latency=true "
        f"! rtph265depay ! h265parse {decode}"
        f"! {sink}"
    )


def run_h265_udp_receiver(args: argparse.Namespace) -> int:
    if args.h265_backend == "gst-launch":
        if shutil.which("gst-launch-1.0") is None:
            print("gst-launch-1.0 not found.")
            return 1
        pipeline = build_h265_udp_pipeline(args, for_opencv=False)
        debug_prefix = f"GST_DEBUG={args.gst_debug} " if args.gst_debug else ""
        cmd = f"{debug_prefix}gst-launch-1.0 -v {pipeline}"
        print("Running:")
        print(cmd)
        return subprocess.call(cmd, shell=True)

    pipeline = build_h265_udp_pipeline(args, for_opencv=True)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Failed to open H.265 UDP receiver. Check OpenCV GStreamer and codec plugins.")
        print(f"Pipeline: {pipeline}")
        print("Try the fallback display backend:")
        print(
            "python3 scripts/visualization/receive_realsense_images.py "
            f"--mode h265_udp --h265-port {args.h265_port} --h265-backend gst-launch"
        )
        return 1

    print(f"Receiving H.265 UDP stream on port {args.h265_port}. Press q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        if not args.no_display:
            cv2.imshow("realsense_h265_receiver", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Receive images from realsense_camera.py")
    parser.add_argument("--mode", choices=["ros2", "h265_udp"], default="ros2")
    parser.add_argument("--rgb-topic", default="rgb/image_compressed")
    parser.add_argument("--rgb-transport", choices=["raw", "compressed", "none"], default="compressed")
    parser.add_argument("--depth-topic", default="depth/image_raw")
    parser.add_argument("--depth-transport", choices=["raw", "compressed", "none"], default="raw")
    parser.add_argument("--display-fps", type=float, default=30.0)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--save-dir", default="")
    parser.add_argument("--save-every", type=int, default=0, help="Save every N received frames; 0 disables saving.")
    parser.add_argument("--video-out", default="", help="Optional path for recording the preview mosaic.")
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-codec", default="mp4v")
    parser.add_argument("--h265-port", type=int, default=5600)
    parser.add_argument("--h265-payload", type=int, default=96)
    parser.add_argument("--h265-latency-ms", type=int, default=0)
    parser.add_argument("--h265-backend", choices=["gst-launch", "opencv"], default="gst-launch")
    parser.add_argument("--h265-sink", choices=["auto", "fps", "fake"], default="fps")
    parser.add_argument("--gst-debug", default="", help="Optional GStreamer debug level, e.g. 2 or rtph265depay:6.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "h265_udp":
        return run_h265_udp_receiver(args)

    rclpy.init()
    node = RealSenseImageReceiver(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
