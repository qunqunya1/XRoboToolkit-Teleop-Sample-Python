import os
import re
import shutil
import subprocess
import threading
import time
from urllib.parse import urlparse

import cv2
import numpy as np

from ...utils.image_utils import compress_image_to_jpg
from .base_camera import BaseCameraInterface


def _build_h265_udp_command(
    port: int,
    width: int,
    height: int,
    payload: int = 96,
    latency_ms: int = 50,
) -> list[str]:
    return [
        "gst-launch-1.0",
        "-q",
        "-e",
        "udpsrc",
        f"port={int(port)}",
        f"caps=application/x-rtp, media=video, encoding-name=H265, payload={int(payload)}, clock-rate=90000",
        "!",
        "rtpjitterbuffer",
        f"latency={int(latency_ms)}",
        "drop-on-latency=true",
        "!",
        "rtph265depay",
        "!",
        "h265parse",
        "!",
        "avdec_h265",
        "!",
        "videoconvert",
        "!",
        "videoscale",
        "!",
        f"video/x-raw,format=BGR,width={int(width)},height={int(height)}",
        "!",
        "fdsink",
        "fd=1",
        "sync=false",
    ]


def _parse_stream_port(uri: str) -> int | None:
    value = str(uri).strip()
    if not value:
        raise ValueError("Video stream URI must be non-empty.")

    if value.isdigit():
        return int(value)

    if re.fullmatch(r":\d+", value):
        return int(value[1:])

    if value.startswith("udp://") or value.startswith("h265_udp://"):
        parsed = urlparse(value.replace("h265_udp://", "udp://", 1))
        if parsed.port is None:
            raise ValueError(f"Video stream URI '{uri}' must include a UDP port.")
        return int(parsed.port)

    return None


class VideoStreamCameraInterface(BaseCameraInterface):
    """Camera interface backed by gst-launch H.265 RTP/UDP receiver subprocesses."""

    def __init__(
        self,
        camera_streams: dict[str, str],
        width: int | None = None,
        height: int | None = None,
        enable_compression: bool = True,
        jpg_quality: int = 85,
        h265_payload: int = 96,
        h265_latency_ms: int = 50,
    ):
        super().__init__(enable_compression=enable_compression, jpg_quality=jpg_quality)
        self.camera_streams = dict(camera_streams)
        self.expected_camera_names = set(self.camera_streams.keys())
        self.width = int(width) if width and int(width) > 0 else None
        self.height = int(height) if height and int(height) > 0 else None
        self.h265_payload = int(h265_payload)
        self.h265_latency_ms = int(h265_latency_ms)
        if self.width is None or self.height is None:
            raise ValueError("Video stream collection requires fixed camera_width and camera_height.")
        if shutil.which("gst-launch-1.0") is None:
            raise RuntimeError("gst-launch-1.0 is not installed or not in PATH.")

        self.processes: dict[str, subprocess.Popen] = {}
        self.reader_threads: dict[str, threading.Thread] = {}
        self.frames_dict: dict[str, dict[str, np.ndarray]] = {}
        self.frames_lock = threading.Lock()
        self.process_lock = threading.Lock()
        self.stop_event = threading.Event()
        self._last_open_attempt: dict[str, float] = {}
        self.frame_nbytes = int(self.width) * int(self.height) * 3

    def start(self):
        self.stop_event.clear()
        with self.process_lock:
            for camera_name in self.camera_streams:
                self._open_process(camera_name)

    def stop(self):
        self.stop_event.set()
        with self.process_lock:
            for process in self.processes.values():
                self._terminate_process(process)
            self.processes = {}
        for thread in self.reader_threads.values():
            thread.join(timeout=1.0)
        self.reader_threads = {}

    @staticmethod
    def _terminate_process(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1.0)

    def _open_process(self, camera_name: str) -> bool:
        now = time.time()
        if now - self._last_open_attempt.get(camera_name, 0.0) < 1.0:
            return False
        self._last_open_attempt[camera_name] = now

        uri = self.camera_streams[camera_name]
        port = _parse_stream_port(uri)
        if port is None:
            print(f"Warning: unsupported video stream URI for '{camera_name}': {uri}")
            return False

        cmd = _build_h265_udp_command(
            port=port,
            width=self.width,
            height=self.height,
            payload=self.h265_payload,
            latency_ms=self.h265_latency_ms,
        )
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except OSError as exc:
            print(f"Warning: failed to start gst-launch for '{camera_name}': {exc}")
            print(f"  Command: {' '.join(cmd)}")
            return False

        if process.stdout is None:
            self._terminate_process(process)
            return False
        self.processes[camera_name] = process
        thread = threading.Thread(
            target=self._reader_loop,
            args=(camera_name, process),
            name=f"{camera_name}_h265_reader",
            daemon=True,
        )
        self.reader_threads[camera_name] = thread
        thread.start()
        return True

    def _reader_loop(self, camera_name: str, process: subprocess.Popen) -> None:
        if process.stdout is None:
            return

        buffer = bytearray()
        fd = process.stdout.fileno()
        while not self.stop_event.is_set() and process.poll() is None:
            try:
                chunk = os.read(fd, self.frame_nbytes)
            except OSError:
                break
            if not chunk:
                break

            buffer.extend(chunk)
            if len(buffer) < self.frame_nbytes:
                continue

            frame_count = len(buffer) // self.frame_nbytes
            start = (frame_count - 1) * self.frame_nbytes
            frame_bytes = bytes(buffer[start : start + self.frame_nbytes])
            del buffer[: frame_count * self.frame_nbytes]
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
                (self.height, self.width, 3)
            )
            with self.frames_lock:
                self.frames_dict[camera_name] = {"color": frame.copy()}

    def update_frames(self):
        with self.process_lock:
            for camera_name in self.camera_streams:
                process = self.processes.get(camera_name)
                if process is None or process.poll() is not None or process.stdout is None:
                    if process is not None:
                        self._terminate_process(process)
                    self.processes.pop(camera_name, None)
                    self._open_process(camera_name)
                    process = self.processes.get(camera_name)
                    if process is None or process.stdout is None:
                        continue

    def get_frames(self):
        with self.frames_lock:
            return {
                camera_name: {"color": data["color"].copy(), "depth": None}
                for camera_name, data in self.frames_dict.items()
                if data.get("color") is not None
            }

    def get_frame(self, identifier: str):
        with self.frames_lock:
            frame = self.frames_dict.get(identifier, {}).get("color")
            return {"color": frame.copy() if frame is not None else None, "depth": None}

    def get_compressed_frames(self):
        with self.frames_lock:
            return {
                camera_name: {
                    "color": compress_image_to_jpg(data.get("color"), self.jpg_quality),
                    "depth": None,
                }
                for camera_name, data in self.frames_dict.items()
                if data.get("color") is not None
            }
