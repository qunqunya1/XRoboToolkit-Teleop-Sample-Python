#!/usr/bin/env python3
"""
RealSense D405 低延迟采集节点（ROS2 / rclpy）

功能：
1) 发布 RGB 原图、压缩图，或 H.265 RTP/UDP 视频流
2) 可选发布深度原图或压缩图（sensor_msgs/Image 或 CompressedImage）

说明：D405 通常没有 RGB 传感器。
若无 RGB 传感器，节点会自动将红外图转成伪 RGB（三通道）后发布到 RGB 话题，
便于统一数据采集流程。
"""

from __future__ import annotations

import threading
import time
import subprocess
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image


class RealSenseImagePublisher(Node):
	def __init__(self) -> None:
		super().__init__("realsense_d405_image_publisher")

		# -------- 参数 --------
		self.declare_parameter("serial_no", "")
		self.declare_parameter("rgb_topic", "rgb/image_raw")
		self.declare_parameter("rgb_compressed_topic", "rgb/image_compressed")
		self.declare_parameter("depth_topic", "depth/image_raw")
		self.declare_parameter("depth_compressed_topic", "depth/image_compressed")
		self.declare_parameter("rgb_transport", "compressed")  # raw / compressed / both / h265_udp
		self.declare_parameter("depth_transport", "raw")  # raw / compressed / both / none
		self.declare_parameter("rgb_jpeg_quality", 80)
		self.declare_parameter("depth_png_compression", 3)
		self.declare_parameter("h265_host", "10.0.1.2")
		self.declare_parameter("h265_port", 5600)
		self.declare_parameter("h265_bitrate_kbps", 2500)
		self.declare_parameter("h265_keyframe_interval", 30)
		self.declare_parameter("h265_encoder", "x265enc")
		self.declare_parameter("h265_backend", "auto")  # auto / opencv / gst_subprocess
		self.declare_parameter("width", 1280)
		self.declare_parameter("height", 720)
		self.declare_parameter("fps", 30)
		self.declare_parameter("infra_index", 1)  # D405 常用 1 或 2
		self.declare_parameter("align_depth_to_rgb", False)
		self.declare_parameter("show_preview", False)
		self.declare_parameter("auto_recover", True)
		self.declare_parameter("degrade_fps_on_timeout", False)
		self.declare_parameter("usb2_safe_mode", False)

		serial_no = self.get_parameter("serial_no").get_parameter_value().string_value
		rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
		rgb_compressed_topic = (
			self.get_parameter("rgb_compressed_topic").get_parameter_value().string_value
		)
		depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
		depth_compressed_topic = (
			self.get_parameter("depth_compressed_topic").get_parameter_value().string_value
		)
		self.rgb_transport = (
			self.get_parameter("rgb_transport").get_parameter_value().string_value.lower()
		)
		self.depth_transport = (
			self.get_parameter("depth_transport").get_parameter_value().string_value.lower()
		)
		self.rgb_jpeg_quality = int(
			self.get_parameter("rgb_jpeg_quality").get_parameter_value().integer_value
		)
		self.depth_png_compression = int(
			self.get_parameter("depth_png_compression").get_parameter_value().integer_value
		)
		self.h265_host = (
			self.get_parameter("h265_host").get_parameter_value().string_value
		)
		self.h265_port = int(
			self.get_parameter("h265_port").get_parameter_value().integer_value
		)
		self.h265_bitrate_kbps = int(
			self.get_parameter("h265_bitrate_kbps").get_parameter_value().integer_value
		)
		self.h265_keyframe_interval = int(
			self.get_parameter("h265_keyframe_interval").get_parameter_value().integer_value
		)
		self.h265_encoder = (
			self.get_parameter("h265_encoder").get_parameter_value().string_value.strip()
		)
		self.h265_backend = (
			self.get_parameter("h265_backend").get_parameter_value().string_value.lower()
		)
		width = self.get_parameter("width").get_parameter_value().integer_value
		height = self.get_parameter("height").get_parameter_value().integer_value
		fps = self.get_parameter("fps").get_parameter_value().integer_value
		infra_index = (
			self.get_parameter("infra_index").get_parameter_value().integer_value
		)
		self.align_depth_to_rgb = (
			self.get_parameter("align_depth_to_rgb").get_parameter_value().bool_value
		)
		self.show_preview = (
			self.get_parameter("show_preview").get_parameter_value().bool_value
		)
		self.auto_recover = (
			self.get_parameter("auto_recover").get_parameter_value().bool_value
		)
		self.degrade_fps_on_timeout = (
			self.get_parameter("degrade_fps_on_timeout").get_parameter_value().bool_value
		)
		self.usb2_safe_mode = (
			self.get_parameter("usb2_safe_mode").get_parameter_value().bool_value
		)
		if self.rgb_transport not in ("raw", "compressed", "both", "h265_udp"):
			self.get_logger().warn(
				f"rgb_transport={self.rgb_transport} 无效，已回退到 compressed"
			)
			self.rgb_transport = "compressed"
		if self.depth_transport not in ("raw", "compressed", "both", "none"):
			self.get_logger().warn(
				f"depth_transport={self.depth_transport} 无效，已回退到 raw"
			)
			self.depth_transport = "raw"
		self.rgb_jpeg_quality = min(100, max(1, self.rgb_jpeg_quality))
		self.depth_png_compression = min(9, max(0, self.depth_png_compression))
		self.h265_port = min(65535, max(1, self.h265_port))
		self.h265_bitrate_kbps = max(100, self.h265_bitrate_kbps)
		self.h265_keyframe_interval = max(1, self.h265_keyframe_interval)
		if self.h265_backend not in ("auto", "opencv", "gst_subprocess"):
			self.get_logger().warn(
				f"h265_backend={self.h265_backend} 无效，已回退到 auto"
			)
			self.h265_backend = "auto"
		self.capture_depth = self.depth_transport != "none"

		# 低延迟 QoS：不追求可靠重传，队列只保留最新一帧
		qos = QoSProfile(
			reliability=ReliabilityPolicy.BEST_EFFORT,
			history=HistoryPolicy.KEEP_LAST,
			depth=1,
		)
		self.rgb_pub = (
			self.create_publisher(Image, rgb_topic, qos)
			if self.rgb_transport in ("raw", "both")
			else None
		)
		self.rgb_compressed_pub = (
			self.create_publisher(CompressedImage, rgb_compressed_topic, qos)
			if self.rgb_transport in ("compressed", "both")
			else None
		)
		self.depth_pub = (
			self.create_publisher(Image, depth_topic, qos)
			if self.depth_transport in ("raw", "both")
			else None
		)
		self.depth_compressed_pub = (
			self.create_publisher(CompressedImage, depth_compressed_topic, qos)
			if self.depth_transport in ("compressed", "both")
			else None
		)
		self.h265_writer: Optional[cv2.VideoWriter] = None
		self.h265_process: Optional[subprocess.Popen] = None

		self.pipeline: Optional[rs.pipeline] = None
		self.aligner: Optional[rs.align] = None
		self.running = True
		self.has_color_sensor = False
		self.usb_type = "unknown"
		self.infra_index = infra_index
		self.preview_lock = threading.Lock()
		self.latest_rgb_preview: Optional[np.ndarray] = None
		self.latest_depth_preview: Optional[np.ndarray] = None
		self.rx_frames = 0
		self.pub_frames = 0
		self.last_frame_time = 0.0
		self.timeout_count = 0
		self.last_timeout_log_time = 0.0
		self.last_restart_attempt_time = 0.0
		self.serial_no = serial_no
		self.width = width
		self.height = height
		self.fps = fps
		self.fps_levels = [self.fps]
		# 去重并保持顺序
		_seen = set()
		self.fps_levels = [x for x in self.fps_levels if not (x in _seen or _seen.add(x))]
		self.fps_index = 0
		self.current_fps = self.fps_levels[self.fps_index]
		self.pipeline_started = False

		# -------- RealSense 初始化 --------
		self.has_color_sensor = self._detect_color_sensor(serial_no)
		self.usb_type = self._detect_usb_type(serial_no)

		if self.usb2_safe_mode and self.usb_type.startswith("2"):
			if (self.width, self.height, self.fps) != (424, 240, 15):
				self.get_logger().warn(
					"检测到 USB2 链路，自动切换到安全配置 424x240@15；"
					"建议改用 USB3 线缆/接口以恢复高帧率。"
				)
			self.width = 424
			self.height = 240
			self.fps = 15
			self.fps_levels = [15]
			self.fps_index = 0
			self.current_fps = 15
		self.pipeline = rs.pipeline()
		config = rs.config()

		if serial_no:
			config.enable_device(serial_no)

		if self.capture_depth:
			config.enable_stream(
				rs.stream.depth,
				self.width,
				self.height,
				rs.format.z16,
				self.current_fps,
			)

		# RGB 流：优先真正的 RGB/Color Camera；D405 的 Stereo Module 走红外伪 RGB
		if self.has_color_sensor:
			config.enable_stream(
				rs.stream.color,
				self.width,
				self.height,
				rs.format.bgr8,
				self.current_fps,
			)
		else:
			config.enable_stream(
				rs.stream.infrared,
				infra_index,
				self.width,
				self.height,
				rs.format.y8,
				self.current_fps,
			)

		profile = self.pipeline.start(config)
		self.pipeline_started = True

		if self.capture_depth and self.align_depth_to_rgb and self.has_color_sensor:
			self.aligner = rs.align(rs.stream.color)

		if self.rgb_transport == "h265_udp":
			self._open_h265_writer()

		# 把设备内部队列设小，减少滞后（某些固件可能不支持，失败则忽略）
		try:
			dev = profile.get_device()
			for sensor in dev.query_sensors():
				if sensor.supports(rs.option.frames_queue_size):
					sensor.set_option(rs.option.frames_queue_size, 1)
		except Exception as e:  # noqa: BLE001
			self.get_logger().warn(f"设置 frames_queue_size 失败，继续运行: {e}")

		rgb_src = "color" if self.has_color_sensor else f"infrared{infra_index}(pseudo-rgb)"
		depth_src = (
			f"z16 {self.width}x{self.height}@{self.current_fps}"
			if self.capture_depth
			else "disabled"
		)
		self.get_logger().info(
			f"Started D405 streams: rgb={rgb_src}, depth={depth_src}, "
			f"rgb_transport={self.rgb_transport}, rgb_topic={rgb_topic}, rgb_compressed_topic={rgb_compressed_topic}, "
			f"depth_transport={self.depth_transport}, depth_topic={depth_topic}, "
			f"depth_compressed_topic={depth_compressed_topic}, align={self.align_depth_to_rgb}, "
			f"h265={self.h265_host}:{self.h265_port}@{self.h265_bitrate_kbps}kbps, "
			f"preview={self.show_preview}, auto_recover={self.auto_recover}, usb_type={self.usb_type}"
		)

		# 独立采集线程，避免堵塞 ROS 回调
		self.worker = threading.Thread(target=self._capture_loop, daemon=True)
		self.worker.start()
		if self.show_preview:
			self.preview_timer = self.create_timer(0.03, self._preview_timer_cb)
		self.status_timer = self.create_timer(1.0, self._status_timer_cb)

	def _detect_color_sensor(self, serial_no: str) -> bool:
		try:
			ctx = rs.context()
			devices = ctx.query_devices()
			if len(devices) == 0:
				self.get_logger().warn("未检测到 RealSense 设备")
				return False

			target = None
			for dev in devices:
				dev_serial = dev.get_info(rs.camera_info.serial_number)
				if not serial_no or serial_no == dev_serial:
					target = dev
					break

			if target is None:
				self.get_logger().warn(
					f"未找到 serial_no={serial_no} 对应设备，按无RGB处理"
				)
				return False

			sensor_names = []
			has_color_stream = False
			for sensor in target.query_sensors():
				name = sensor.get_info(rs.camera_info.name)
				sensor_names.append(name)
				name_lower = name.lower()
				is_dedicated_color_sensor = (
					"rgb" in name_lower or "color camera" in name_lower
				)
				if not is_dedicated_color_sensor:
					continue

				for p in sensor.get_stream_profiles():
					try:
						if p.stream_type() == rs.stream.color:
							has_color_stream = True
							break
					except Exception:
						continue

			if has_color_stream:
				self.get_logger().info(
					f"检测到彩色流支持，sensors={sensor_names}"
				)
			else:
				self.get_logger().warn(
					f"未检测到独立 RGB/Color Camera，sensors={sensor_names}，将使用红外伪RGB"
				)
			return has_color_stream
		except Exception as e:  # noqa: BLE001
			self.get_logger().warn(f"检测 RGB 传感器失败，按无RGB处理: {e}")

		return False

	def _detect_usb_type(self, serial_no: str) -> str:
		try:
			ctx = rs.context()
			devices = ctx.query_devices()
			if len(devices) == 0:
				return "unknown"

			target = None
			for dev in devices:
				dev_serial = dev.get_info(rs.camera_info.serial_number)
				if not serial_no or serial_no == dev_serial:
					target = dev
					break

			if target is None:
				return "unknown"

			if target.supports(rs.camera_info.usb_type_descriptor):
				return target.get_info(rs.camera_info.usb_type_descriptor)
		except Exception:
			pass
		return "unknown"

	def _capture_loop(self) -> None:
		while rclpy.ok() and self.running:
			try:
				if not self.pipeline_started:
					now = time.time()
					if self.auto_recover and (now - self.last_restart_attempt_time) > 1.5:
						self._restart_pipeline(degrade=False)
					time.sleep(0.05)
					continue

				# 兼顾低延迟和低FPS模式稳定性
				timeout_ms = max(120, int(2000 / max(1, self.current_fps)))
				frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
				self.timeout_count = 0
				if self.aligner is not None:
					frames = self.aligner.process(frames)

				depth_frame = None
				if self.capture_depth:
					depth_frame = frames.get_depth_frame()
					if not depth_frame:
						continue

				if self.has_color_sensor:
					rgb_frame = frames.get_color_frame()
					if not rgb_frame:
						continue
					rgb_img = np.asanyarray(rgb_frame.get_data())  # BGR8
				else:
					ir = frames.get_infrared_frame(self.infra_index)
					if not ir:
						ir = frames.get_infrared_frame()
					if not ir:
						continue
					gray = np.asanyarray(ir.get_data())
					rgb_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

				self.rx_frames += 1
				self.last_frame_time = time.time()

				depth_img = None
				if depth_frame is not None:
					depth_img = np.asanyarray(depth_frame.get_data())  # uint16, 16UC1
					if depth_img.dtype != np.uint16:
						depth_img = depth_img.astype(np.uint16)

				if self.show_preview:
					with self.preview_lock:
						self.latest_rgb_preview = rgb_img.copy()
						self.latest_depth_preview = (
							depth_img.copy() if depth_img is not None else None
						)

				stamp = self.get_clock().now().to_msg()

				published = False
				if self.rgb_pub is not None:
					self.rgb_pub.publish(self._make_rgb_msg(rgb_img, stamp))
					published = True
				if self.rgb_compressed_pub is not None:
					msg = self._make_compressed_rgb_msg(rgb_img, stamp)
					if msg is not None:
						self.rgb_compressed_pub.publish(msg)
						published = True
				if self.h265_writer is not None:
					self.h265_writer.write(rgb_img)
					published = True
				if self.h265_process is not None and self.h265_process.stdin is not None:
					try:
						self.h265_process.stdin.write(rgb_img.tobytes())
						published = True
					except BrokenPipeError:
						self.get_logger().error("H.265 gst-launch 子进程管道断开")
						self._close_h265_process()

				if depth_img is not None and self.depth_pub is not None:
					self.depth_pub.publish(self._make_depth_msg(depth_img, stamp))
					published = True
				if depth_img is not None and self.depth_compressed_pub is not None:
					msg = self._make_compressed_depth_msg(depth_img, stamp)
					if msg is not None:
						self.depth_compressed_pub.publish(msg)
						published = True

				if published:
					self.pub_frames += 1

			except RuntimeError as e:
				# 常见于超时或设备瞬时不可用
				err = str(e)
				if "context is invalid" in err or not rclpy.ok() or not self.running:
					break

				self.timeout_count += 1
				now = time.time()
				if now - self.last_timeout_log_time > 1.0:
					self.last_timeout_log_time = now
					self.get_logger().warn(
						f"wait_for_frames 超时/异常: {err} (timeout_count={self.timeout_count})"
					)

				if "cannot be called before start" in err:
					self.pipeline_started = False

				if self.auto_recover and self.timeout_count >= 20:
					self.get_logger().warn("连续超时，尝试重启 RealSense pipeline...")
					self._restart_pipeline(degrade=self.degrade_fps_on_timeout)
					self.timeout_count = 0
				continue
			except Exception as e:  # noqa: BLE001
				if not rclpy.ok() or not self.running:
					break
				self.get_logger().error(f"采集线程异常: {e}")

	def _build_h265_pipeline(self) -> str:
		if self.h265_encoder == "x265enc":
			encoder = (
				f"x265enc tune=zerolatency bitrate={self.h265_bitrate_kbps} "
				f"key-int-max={self.h265_keyframe_interval} speed-preset=ultrafast"
			)
		else:
			encoder = self.h265_encoder

		return (
			"appsrc is-live=true block=false format=time do-timestamp=true "
			f"! video/x-raw,format=BGR,width={self.width},height={self.height},"
			f"framerate={self.current_fps}/1 "
			"! queue leaky=downstream max-size-buffers=1 "
			"! videoconvert "
			"! video/x-raw,format=I420 "
			f"! {encoder} "
			"! h265parse config-interval=1 "
			"! rtph265pay pt=96 "
			f"! udpsink host={self.h265_host} port={self.h265_port} "
			"sync=false async=false"
		)

	def _build_h265_subprocess_cmd(self) -> list[str]:
		if self.h265_encoder == "x265enc":
			encoder = [
				"x265enc",
				"tune=zerolatency",
				f"bitrate={self.h265_bitrate_kbps}",
				f"key-int-max={self.h265_keyframe_interval}",
				"speed-preset=ultrafast",
			]
		else:
			encoder = self.h265_encoder.split()

		return [
			"gst-launch-1.0",
			"-q",
			"fdsrc",
			"fd=0",
			"do-timestamp=true",
			"!",
			"rawvideoparse",
			"format=bgr",
			f"width={self.width}",
			f"height={self.height}",
			f"framerate={self.current_fps}/1",
			"!",
			"queue",
			"leaky=downstream",
			"max-size-buffers=1",
			"!",
			"videoconvert",
			"!",
			"video/x-raw,format=I420",
			"!",
			*encoder,
			"!",
			"h265parse",
			"config-interval=1",
			"!",
			"rtph265pay",
			"pt=96",
			"!",
			"udpsink",
			f"host={self.h265_host}",
			f"port={self.h265_port}",
			"sync=false",
			"async=false",
		]

	def _open_h265_writer(self) -> None:
		self._close_h265_writer()
		self._close_h265_process()

		if self.h265_backend == "gst_subprocess":
			self._open_h265_process()
			return

		pipeline = self._build_h265_pipeline()
		self.get_logger().info(f"H.265 pipeline: {pipeline}")
		writer = cv2.VideoWriter(
			pipeline,
			cv2.CAP_GSTREAMER,
			0,
			float(self.current_fps),
			(int(self.width), int(self.height)),
			True,
		)
		if not writer.isOpened():
			self.get_logger().error(
				"H.265 视频流启动失败：请确认 OpenCV 启用了 GStreamer，"
				"并已安装 H.265 编码/解析插件，例如 x265enc、h265parse、rtph265pay。"
			)
			writer.release()
			self.h265_writer = None
			if self.h265_backend == "auto":
				self.get_logger().warn("切换到 gst-launch 子进程方式发送 H.265")
				self._open_h265_process()
			return

		self.h265_writer = writer
		self.get_logger().info(
			f"H.265 RTP/UDP 视频流已启动: udp://{self.h265_host}:{self.h265_port}"
		)

	def _open_h265_process(self) -> None:
		cmd = self._build_h265_subprocess_cmd()
		self.get_logger().info(f"H.265 gst-launch command: {' '.join(cmd)}")
		try:
			self.h265_process = subprocess.Popen(
				cmd,
				stdin=subprocess.PIPE,
				stdout=subprocess.DEVNULL,
				stderr=subprocess.PIPE,
			)
		except Exception as e:  # noqa: BLE001
			self.h265_process = None
			self.get_logger().error(f"启动 gst-launch H.265 子进程失败: {e}")
			return

		self.get_logger().info(
			f"H.265 RTP/UDP 视频流已启动(gst-launch): udp://{self.h265_host}:{self.h265_port}"
		)

	def _close_h265_writer(self) -> None:
		if self.h265_writer is not None:
			try:
				self.h265_writer.release()
			except Exception:
				pass
			self.h265_writer = None

	def _close_h265_process(self) -> None:
		if self.h265_process is None:
			return
		try:
			if self.h265_process.stdin is not None:
				self.h265_process.stdin.close()
			self.h265_process.terminate()
			self.h265_process.wait(timeout=1.0)
		except Exception:
			try:
				self.h265_process.kill()
			except Exception:
				pass
		self.h265_process = None

	def _make_rgb_msg(self, rgb_img: np.ndarray, stamp) -> Image:
		msg = Image()
		msg.header.stamp = stamp
		msg.header.frame_id = "d405_rgb"
		msg.height = int(rgb_img.shape[0])
		msg.width = int(rgb_img.shape[1])
		msg.encoding = "bgr8"
		msg.is_bigendian = False
		msg.step = int(rgb_img.shape[1] * rgb_img.shape[2])
		msg.data = rgb_img.tobytes()
		return msg

	def _make_depth_msg(self, depth_img: np.ndarray, stamp) -> Image:
		msg = Image()
		msg.header.stamp = stamp
		msg.header.frame_id = "d405_depth"
		msg.height = int(depth_img.shape[0])
		msg.width = int(depth_img.shape[1])
		msg.encoding = "16UC1"
		msg.is_bigendian = False
		msg.step = int(depth_img.shape[1] * 2)
		msg.data = depth_img.tobytes()
		return msg

	def _make_compressed_rgb_msg(
		self, rgb_img: np.ndarray, stamp
	) -> Optional[CompressedImage]:
		ok, encoded = cv2.imencode(
			".jpg",
			rgb_img,
			[int(cv2.IMWRITE_JPEG_QUALITY), self.rgb_jpeg_quality],
		)
		if not ok:
			self.get_logger().warn("RGB JPEG 压缩失败，跳过该帧")
			return None

		msg = CompressedImage()
		msg.header.stamp = stamp
		msg.header.frame_id = "d405_rgb"
		msg.format = "bgr8; jpeg compressed bgr8"
		msg.data = encoded.tobytes()
		return msg

	def _make_compressed_depth_msg(
		self, depth_img: np.ndarray, stamp
	) -> Optional[CompressedImage]:
		ok, encoded = cv2.imencode(
			".png",
			depth_img,
			[int(cv2.IMWRITE_PNG_COMPRESSION), self.depth_png_compression],
		)
		if not ok:
			self.get_logger().warn("Depth PNG 压缩失败，跳过该帧")
			return None

		msg = CompressedImage()
		msg.header.stamp = stamp
		msg.header.frame_id = "d405_depth"
		msg.format = "16UC1; png compressed 16UC1"
		msg.data = encoded.tobytes()
		return msg

	def _restart_pipeline(self, degrade: bool = False) -> None:
		self.last_restart_attempt_time = time.time()
		try:
			if self.pipeline is not None:
				self.pipeline.stop()
		except Exception:
			pass

		try:
			if degrade and self.fps_index < len(self.fps_levels) - 1:
				self.fps_index += 1
				self.current_fps = self.fps_levels[self.fps_index]
				self.get_logger().warn(f"降帧恢复: 切换到 {self.current_fps} FPS")

			self.pipeline = rs.pipeline()
			config = rs.config()
			if self.serial_no:
				config.enable_device(self.serial_no)

			config.enable_stream(
				rs.stream.depth,
				self.width,
				self.height,
				rs.format.z16,
				self.current_fps,
			)

			if self.has_color_sensor:
				config.enable_stream(
					rs.stream.color,
					self.width,
					self.height,
					rs.format.bgr8,
					self.current_fps,
				)
			else:
				config.enable_stream(
					rs.stream.infrared,
					self.infra_index,
					self.width,
					self.height,
					rs.format.y8,
					self.current_fps,
				)

			profile = self.pipeline.start(config)
			self.pipeline_started = True
			if self.capture_depth and self.align_depth_to_rgb and self.has_color_sensor:
				self.aligner = rs.align(rs.stream.color)
			else:
				self.aligner = None

			if self.rgb_transport == "h265_udp":
				self._open_h265_writer()

			try:
				dev = profile.get_device()
				for sensor in dev.query_sensors():
					if sensor.supports(rs.option.frames_queue_size):
						sensor.set_option(rs.option.frames_queue_size, 1)
			except Exception:
				pass

			self.get_logger().info(
				f"RealSense pipeline 重启成功，当前配置: {self.width}x{self.height}@{self.current_fps}"
			)
		except Exception as e:  # noqa: BLE001
			self.pipeline_started = False
			self.get_logger().error(f"RealSense pipeline 重启失败: {e}")

	def _status_timer_cb(self) -> None:
		now = time.time()
		if self.last_frame_time <= 0:
			self.get_logger().warn("状态: 尚未接收到相机帧")
			return

		gap = now - self.last_frame_time
		if gap > 1.5:
			self.get_logger().warn(
				f"状态: 最近 {gap:.2f}s 未接收到新帧, rx={self.rx_frames}, pub={self.pub_frames}"
			)
		else:
			self.get_logger().info(
				f"状态: 正常接收中, rx={self.rx_frames}, pub={self.pub_frames}, "
				f"last_gap={gap*1000:.0f}ms, cfg={self.width}x{self.height}@{self.current_fps}"
			)

	def _preview_timer_cb(self) -> None:
		if not self.show_preview:
			return
		with self.preview_lock:
			if self.latest_rgb_preview is None:
				return
			rgb = self.latest_rgb_preview
			depth = self.latest_depth_preview
		self._show_preview(rgb, depth)

	def _show_preview(
		self, rgb_img: np.ndarray, depth_img: Optional[np.ndarray]
	) -> None:
		try:
			cv2.imshow("publisher_rgb", rgb_img)
			if depth_img is not None:
				depth_vis = cv2.convertScaleAbs(depth_img, alpha=255.0 / max(1, int(depth_img.max())))
				depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
				cv2.imshow("publisher_depth", depth_vis)
			cv2.waitKey(1)
		except Exception as e:  # noqa: BLE001
			self.show_preview = False
			self.get_logger().warn(f"预览窗口不可用，已自动关闭预览: {e}")

	def destroy_node(self) -> bool:
		self.running = False
		if hasattr(self, "worker") and self.worker.is_alive():
			self.worker.join(timeout=1.0)
		if self.pipeline is not None:
			try:
				self.pipeline.stop()
			except Exception:  # noqa: BLE001
				pass
		self._close_h265_writer()
		if self.show_preview:
			try:
				cv2.destroyAllWindows()
			except Exception:  # noqa: BLE001
				pass
		return super().destroy_node()


def main(args=None) -> None:
	rclpy.init(args=args)
	node = RealSenseImagePublisher()
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


if __name__ == "__main__":
	main()
