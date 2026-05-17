#!/usr/bin/env python3
"""
ROS2 订阅节点：
1) 支持双相机（right / left）订阅 RGB/深度原图或压缩图
2) 哪个相机有数据就显示哪个；两个都有则都显示
3) 近似时间同步后可选保存数据（用于采集）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image


@dataclass
class CachedMsg:
    stamp_ns: int
    msg: object


@dataclass
class CamState:
    latest_rgb: Optional[CachedMsg] = None
    latest_depth: Optional[CachedMsg] = None
    rgb_count: int = 0
    depth_count: int = 0
    pair_count: int = 0
    last_rx_time: float = 0.0
    last_status_time: float = 0.0
    last_rgb_count: int = 0
    last_depth_count: int = 0
    rgb_size: Optional[tuple[int, int]] = None
    depth_size: Optional[tuple[int, int]] = None


class RGBDepthSubscriber(Node):
    def __init__(self) -> None:
        super().__init__("rgb_depth_subscriber")

        self.declare_parameter("right_rgb_topic", "right/rgb/image_raw")
        self.declare_parameter("right_depth_topic", "right/depth/image_raw")
        # left 相机可选，默认匹配双相机发布脚本
        self.declare_parameter("left_rgb_topic", "left/rgb/image_raw")
        self.declare_parameter("left_depth_topic", "left/depth/image_raw")
        self.declare_parameter("rgb_transport", "compressed")  # raw / compressed
        self.declare_parameter("depth_transport", "raw")  # raw / compressed
        self.declare_parameter("sync_tolerance_ms", 30)
        self.declare_parameter("show_preview", True)
        self.declare_parameter("save", False)
        self.declare_parameter("save_dir", "./dataset")

        right_rgb_topic = (
            self.get_parameter("right_rgb_topic").get_parameter_value().string_value
        )
        right_depth_topic = (
            self.get_parameter("right_depth_topic").get_parameter_value().string_value
        )
        left_rgb_topic = (
            self.get_parameter("left_rgb_topic").get_parameter_value().string_value
        )
        left_depth_topic = (
            self.get_parameter("left_depth_topic").get_parameter_value().string_value
        )
        self.rgb_transport = (
            self.get_parameter("rgb_transport").get_parameter_value().string_value.lower()
        )
        self.depth_transport = (
            self.get_parameter("depth_transport").get_parameter_value().string_value.lower()
        )
        if self.rgb_transport not in ("raw", "compressed"):
            self.get_logger().warn(
                f"rgb_transport={self.rgb_transport} 无效，已回退到 compressed"
            )
            self.rgb_transport = "compressed"
        if self.depth_transport not in ("raw", "compressed"):
            self.get_logger().warn(
                f"depth_transport={self.depth_transport} 无效，已回退到 raw"
            )
            self.depth_transport = "raw"

        self.sync_tolerance_ns = (
            self.get_parameter("sync_tolerance_ms").get_parameter_value().integer_value
            * 1_000_000
        )
        self.show_preview = (
            self.get_parameter("show_preview").get_parameter_value().bool_value
        )
        self.save = self.get_parameter("save").get_parameter_value().bool_value
        self.save_dir = Path(
            self.get_parameter("save_dir").get_parameter_value().string_value
        )

        self.cams: dict[str, CamState] = {
            "right": CamState(last_rx_time=time.time(), last_status_time=time.time()),
            "left": CamState(last_rx_time=time.time(), last_status_time=time.time()),
        }

        self.enabled_cams: list[str] = []
        self._subscription_refs = []

        if self.save:
            for cam in ("right", "left"):
                (self.save_dir / cam / "rgb").mkdir(parents=True, exist_ok=True)
                (self.save_dir / cam / "depth").mkdir(parents=True, exist_ok=True)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.saved_count = 0

        # right 订阅（必须有）
        self._create_cam_subscriptions("right", right_rgb_topic, right_depth_topic, qos)

        # left 订阅（可选）
        if left_rgb_topic and left_depth_topic:
            self._create_cam_subscriptions("left", left_rgb_topic, left_depth_topic, qos)
        else:
            self.get_logger().info("left 相机话题为空，按单相机模式运行")

        self.status_timer = self.create_timer(2.0, self._status_report)

        self.get_logger().info(
            f"Subscribed cams={self.enabled_cams}, rgb_transport={self.rgb_transport}, "
            f"depth_transport={self.depth_transport}, tol={self.sync_tolerance_ns / 1e6:.1f}ms"
        )

    def _create_cam_subscriptions(
        self,
        cam: str,
        rgb_topic: str,
        depth_topic: str,
        qos: QoSProfile,
    ) -> None:
        rgb_msg_type = CompressedImage if self.rgb_transport == "compressed" else Image
        depth_msg_type = CompressedImage if self.depth_transport == "compressed" else Image
        self._subscription_refs.append(
            self.create_subscription(
                rgb_msg_type,
                rgb_topic,
                lambda msg, c=cam: self._on_rgb(c, msg),
                qos,
            )
        )
        self._subscription_refs.append(
            self.create_subscription(
                depth_msg_type,
                depth_topic,
                lambda msg, c=cam: self._on_depth(c, msg),
                qos,
            )
        )
        self.enabled_cams.append(cam)
        self.get_logger().info(f"{cam}: rgb={rgb_topic}, depth={depth_topic}")

    def _stamp_to_ns(self, sec: int, nanosec: int) -> int:
        return sec * 1_000_000_000 + nanosec

    def _on_rgb(self, cam: str, msg: Image | CompressedImage) -> None:
        ns = self._stamp_to_ns(msg.header.stamp.sec, msg.header.stamp.nanosec)
        st = self.cams[cam]
        st.latest_rgb = CachedMsg(stamp_ns=ns, msg=msg)
        st.rgb_count += 1
        st.last_rx_time = time.time()
        if isinstance(msg, Image):
            st.rgb_size = (msg.width, msg.height)
        self._try_process_pair(cam)

    def _on_depth(self, cam: str, msg: Image | CompressedImage) -> None:
        ns = self._stamp_to_ns(msg.header.stamp.sec, msg.header.stamp.nanosec)
        st = self.cams[cam]
        st.latest_depth = CachedMsg(stamp_ns=ns, msg=msg)
        st.depth_count += 1
        st.last_rx_time = time.time()
        if isinstance(msg, Image):
            st.depth_size = (msg.width, msg.height)
        self._try_process_pair(cam)

    def _try_process_pair(self, cam: str) -> None:
        st = self.cams[cam]
        if st.latest_rgb is None or st.latest_depth is None:
            return

        dt = abs(st.latest_rgb.stamp_ns - st.latest_depth.stamp_ns)
        synced = dt <= self.sync_tolerance_ns

        rgb_msg = st.latest_rgb.msg
        depth_msg = st.latest_depth.msg

        rgb = self._decode_rgb(rgb_msg)
        depth = self._decode_depth(depth_msg)
        if rgb is None or depth is None:
            return
        st.rgb_size = (int(rgb.shape[1]), int(rgb.shape[0]))
        st.depth_size = (int(depth.shape[1]), int(depth.shape[0]))

        if self.show_preview:
            self._show(cam, rgb, depth)

        if synced and self.save:
            ts = min(st.latest_rgb.stamp_ns, st.latest_depth.stamp_ns)
            self._save_pair(cam, rgb, depth, ts)

        if synced:
            st.pair_count += 1

    def _decode_rgb(self, msg: Image | CompressedImage) -> Optional[np.ndarray]:
        if isinstance(msg, CompressedImage):
            data = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                self.get_logger().warn("RGB压缩图解码失败")
            return img

        if msg.encoding not in ("bgr8", "rgb8", "mono8"):
            self.get_logger().warn(f"不支持的RGB编码: {msg.encoding}")
            return None

        channels = 1 if msg.encoding == "mono8" else 3
        img = np.frombuffer(msg.data, dtype=np.uint8)
        if img.size != msg.height * msg.width * channels:
            self.get_logger().warn("RGB数据尺寸不匹配")
            return None

        if channels == 1:
            return cv2.cvtColor(img.reshape((msg.height, msg.width)), cv2.COLOR_GRAY2BGR)

        img = img.reshape((msg.height, msg.width, channels))
        if msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def _decode_depth(self, msg: Image | CompressedImage) -> Optional[np.ndarray]:
        if isinstance(msg, CompressedImage):
            data = np.frombuffer(msg.data, dtype=np.uint8)
            depth = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if depth is None:
                self.get_logger().warn("深度压缩图解码失败")
                return None
            if depth.dtype != np.uint16:
                self.get_logger().warn(f"压缩深度图dtype异常: {depth.dtype}")
                return None
            return depth

        if msg.encoding != "16UC1":
            self.get_logger().warn(f"不支持的深度编码: {msg.encoding}")
            return None

        depth = np.frombuffer(msg.data, dtype=np.uint16)
        if depth.size != msg.height * msg.width:
            self.get_logger().warn("深度数据尺寸不匹配")
            return None

        return depth.reshape((msg.height, msg.width))

    def _show(self, cam: str, rgb: np.ndarray, depth: np.ndarray) -> None:
        depth_vis = cv2.convertScaleAbs(depth, alpha=255.0 / max(1, int(depth.max())))
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.imshow(f"{cam}_rgb", rgb)
        cv2.imshow(f"{cam}_depth", depth_vis)
        cv2.waitKey(1)

    def _save_pair(self, cam: str, rgb: np.ndarray, depth: np.ndarray, ts_ns: int) -> None:
        rgb_path = self.save_dir / cam / "rgb" / f"{ts_ns}.jpg"
        depth_path = self.save_dir / cam / "depth" / f"{ts_ns}.png"

        cv2.imwrite(str(rgb_path), rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        # 16-bit PNG，保留深度精度
        cv2.imwrite(str(depth_path), depth)

        self.saved_count += 1
        if self.saved_count % 30 == 0:
            self.get_logger().info(f"已保存 {self.saved_count} 对数据")

    def _status_report(self) -> None:
        any_data = False
        for cam in self.enabled_cams:
            st = self.cams[cam]
            idle_s = time.time() - st.last_rx_time
            if st.rgb_count > 0 or st.depth_count > 0:
                any_data = True

            if st.rgb_count == 0 and st.depth_count == 0:
                self.get_logger().warn(f"{cam}: 仍未收到任何消息")
                continue

            if idle_s > 2.5:
                self.get_logger().warn(
                    f"{cam}: 最近 {idle_s:.1f}s 未收到新消息。"
                    f"rgb={st.rgb_count}, depth={st.depth_count}, synced={st.pair_count}"
                )
                continue

            now = time.time()
            elapsed = max(1e-6, now - st.last_status_time)
            rgb_fps = (st.rgb_count - st.last_rgb_count) / elapsed
            depth_fps = (st.depth_count - st.last_depth_count) / elapsed
            rgb_size = (
                f"{st.rgb_size[0]}x{st.rgb_size[1]}" if st.rgb_size else "unknown"
            )
            depth_size = (
                f"{st.depth_size[0]}x{st.depth_size[1]}"
                if st.depth_size
                else "unknown"
            )
            self.get_logger().info(
                f"{cam}: rgb_fps={rgb_fps:.1f}, rgb_res={rgb_size}, "
                f"depth_fps={depth_fps:.1f}, depth_res={depth_size}, "
                f"total_rgb={st.rgb_count}, total_depth={st.depth_count}, synced={st.pair_count}"
            )
            st.last_status_time = now
            st.last_rgb_count = st.rgb_count
            st.last_depth_count = st.depth_count

        if not any_data:
            self.get_logger().warn("请确认发布节点正在运行，且在同一 ROS_DOMAIN_ID。")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RGBDepthSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.show_preview:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
