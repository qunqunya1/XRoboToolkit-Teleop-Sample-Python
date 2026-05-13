import argparse
from pathlib import Path
import sys
import time


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


_REPO_ROOT = _find_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


try:
    import rclpy
    from aimdk_msgs.srv import SetVolume
    from rclpy.node import Node
except ImportError as exc:  # pragma: no cover - depends on ROS2 runtime
    rclpy = None
    SetVolume = None
    Node = object
    _ROS2_IMPORT_ERROR = exc
else:
    _ROS2_IMPORT_ERROR = None


DEFAULT_SET_VOLUME_SERVICE_NAME = "/aimdk_5Fmsgs/srv/SetVolume"
DEFAULT_AUDIO_VOLUME = 10


class SetVolumeClient(Node):
    def __init__(self, service_name: str):
        super().__init__("x2_set_volume_client")
        self._client = self.create_client(SetVolume, service_name)
        self._service_name = service_name

    def wait_for_service(self, timeout_s: float) -> bool:
        return self._client.wait_for_service(timeout_sec=timeout_s)

    def call(self, audio_volume: int, timeout_s: float):
        request = SetVolume.Request()
        request.audio_volume = int(audio_volume)
        future = self._client.call_async(request)
        deadline = time.monotonic() + timeout_s

        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for response from service {self._service_name!r}."
                )

        if not future.done():
            raise RuntimeError("ROS2 shutdown before the SetVolume service returned.")

        exception = future.exception()
        if exception is not None:
            raise RuntimeError(f"SetVolume service call failed: {exception}") from exception
        return future.result()


def main(
    audio_volume: int = DEFAULT_AUDIO_VOLUME,
    service_name: str = DEFAULT_SET_VOLUME_SERVICE_NAME,
    service_wait_timeout_s: float = 5.0,
    response_timeout_s: float = 10.0,
):
    """Call the X2 SetVolume ROS2 service to change robot audio volume."""

    if _ROS2_IMPORT_ERROR is not None:
        raise RuntimeError(
            "ROS2 dependencies are unavailable. Please source your ROS2 workspace "
            "so rclpy and the generated aimdk_msgs Python package can be imported."
        ) from _ROS2_IMPORT_ERROR

    if not 0 <= int(audio_volume) <= 100:
        raise ValueError(f"audio_volume must be in [0, 100], got {audio_volume}.")

    if not rclpy.ok():
        rclpy.init(args=None)

    node = SetVolumeClient(service_name=service_name)
    try:
        print(f"Waiting for SetVolume service: {service_name}")
        if not node.wait_for_service(timeout_s=service_wait_timeout_s):
            raise TimeoutError(
                f"SetVolume service {service_name!r} did not become available within "
                f"{service_wait_timeout_s:.1f}s."
            )

        print(f"Sending request: set volume to {int(audio_volume)}")
        response = node.call(audio_volume=audio_volume, timeout_s=response_timeout_s)

        current_volume = int(getattr(response, "audio_volume", audio_volume))
        response_header = getattr(response, "reponse", None)
        response_message = getattr(response_header, "message", "")
        response_status = getattr(response_header, "status", None)

        print("SetVolume succeeded.")
        print(f"Current audio volume: {current_volume}")
        if response_status is not None:
            print(f"Response status: {response_status}")
        if response_message:
            print(f"Response message: {response_message}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Call the X2 SetVolume ROS2 service to change robot audio volume."
    )
    parser.add_argument(
        "--audio-volume",
        type=int,
        default=DEFAULT_AUDIO_VOLUME,
        help="Target audio volume in [0, 100]. Default: 10.",
    )
    parser.add_argument(
        "--service-name",
        default=DEFAULT_SET_VOLUME_SERVICE_NAME,
        help="ROS2 SetVolume service name.",
    )
    parser.add_argument(
        "--service-wait-timeout-s",
        type=float,
        default=5.0,
        help="Seconds to wait for the service to appear.",
    )
    parser.add_argument(
        "--response-timeout-s",
        type=float,
        default=10.0,
        help="Seconds to wait for the service response.",
    )
    args = parser.parse_args()

    main(
        audio_volume=args.audio_volume,
        service_name=args.service_name,
        service_wait_timeout_s=args.service_wait_timeout_s,
        response_timeout_s=args.response_timeout_s,
    )
