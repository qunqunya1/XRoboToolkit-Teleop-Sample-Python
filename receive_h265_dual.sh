#!/usr/bin/env bash
set -eo pipefail

RIGHT_PORT="${RIGHT_PORT:-5600}"
LEFT_PORT="${LEFT_PORT:-5602}"
RECEIVE_RIGHT="${RECEIVE_RIGHT:-true}"
RECEIVE_LEFT="${RECEIVE_LEFT:-true}"
VIDEO_SINK="${VIDEO_SINK:-autovideosink}"
SYNC="${SYNC:-false}"
LATENCY_MS="${LATENCY_MS:-50}"
PAYLOAD_TYPE="${PAYLOAD_TYPE:-96}"
SHOW_VERBOSE="${SHOW_VERBOSE:-false}"
SHOW_CAPS="${SHOW_CAPS:-true}"
SHOW_FPS="${SHOW_FPS:-true}"
FPS_INTERVAL_MS="${FPS_INTERVAL_MS:-1000}"
RECORD="${RECORD:-false}"
RECORD_DIR="${RECORD_DIR:-./recordings}"
RECORD_SECONDS="${RECORD_SECONDS:-0}"
RECORD_CONTAINER="${RECORD_CONTAINER:-mp4}"
RECORD_PREVIEW="${RECORD_PREVIEW:-false}"

usage() {
  cat <<EOF
Usage:
  bash scripts/receive_h265_dual.sh

Environment variables:
  RECEIVE_RIGHT=true|false   Whether to receive right camera. Default: true
  RECEIVE_LEFT=true|false    Whether to receive left camera. Default: true
  RIGHT_PORT=5600            Right camera UDP/RTP port.
  LEFT_PORT=5602             Left camera UDP/RTP port.
  VIDEO_SINK=autovideosink   GStreamer video sink, e.g. xvimagesink, ximagesink.
  SYNC=false                 Use sink clock sync.
  LATENCY_MS=50              rtpjitterbuffer latency in ms.
  PAYLOAD_TYPE=96            RTP payload type.
  SHOW_VERBOSE=false         Use gst-launch -v.
  SHOW_CAPS=true             Print negotiated caps, including width/height.
  SHOW_FPS=true              Print measured render FPS.
  FPS_INTERVAL_MS=1000       FPS print interval in ms.
  RECORD=false               Save received H.265 stream to file without re-encoding.
  RECORD_DIR=./recordings    Output directory.
  RECORD_SECONDS=0           Stop after N seconds; 0 means until Ctrl+C.
  RECORD_CONTAINER=mp4       mp4 or mkv.
  RECORD_PREVIEW=false       Also show preview while recording.

Examples:
  bash scripts/receive_h265_dual.sh
  RECEIVE_LEFT=false RIGHT_PORT=5600 bash scripts/receive_h265_dual.sh
  VIDEO_SINK=xvimagesink bash scripts/receive_h265_dual.sh
  RECORD=true RECORD_SECONDS=10 RECEIVE_LEFT=false bash scripts/receive_h265_dual.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[receive_h265_dual] 缺少命令: $1"
    exit 1
  fi
}

require_element() {
  if ! gst-inspect-1.0 "$1" >/dev/null 2>&1; then
    echo "[receive_h265_dual] 缺少 GStreamer 插件: $1"
    echo "[receive_h265_dual] 可尝试安装：sudo apt install gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav"
    exit 1
  fi
}

require_cmd gst-launch-1.0
require_cmd gst-inspect-1.0
require_element rtph265depay
require_element h265parse
require_element avdec_h265
if [[ "$SHOW_FPS" == "true" ]]; then
  require_element fpsdisplaysink
fi
if [[ "$RECORD" == "true" ]]; then
  require_element filesink
  if [[ "$RECORD_CONTAINER" == "mp4" ]]; then
    require_element mp4mux
  elif [[ "$RECORD_CONTAINER" == "mkv" ]]; then
    require_element matroskamux
  else
    echo "[receive_h265_dual] RECORD_CONTAINER 只支持 mp4 或 mkv"
    exit 1
  fi
  mkdir -p "$RECORD_DIR"
fi

GST_FLAGS=(-q)
if [[ "$SHOW_VERBOSE" == "true" || "$SHOW_CAPS" == "true" ]]; then
  GST_FLAGS=(-v)
fi

PIDS=()
STOPPING=false
cleanup() {
  if [[ "$STOPPING" == "true" ]]; then
    return
  fi
  STOPPING=true
  for p in "${PIDS[@]:-}"; do
    kill -INT "$p" 2>/dev/null || true
  done
  sleep 1
  for p in "${PIDS[@]:-}"; do
    kill -TERM "$p" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

print_recordings() {
  if [[ "$RECORD" != "true" ]]; then
    return
  fi
  echo "[receive_h265_dual] recordings:"
  find "$RECORD_DIR" -maxdepth 1 -type f \( -name "*.mp4" -o -name "*.mkv" \) -printf "  %p  %s bytes\n" 2>/dev/null | sort || true
}

start_receiver() {
  local name="$1"
  local port="$2"

  echo "[receive_h265_dual] starting ${name}: udp://0.0.0.0:${port}"
  if [[ "$RECORD" == "true" ]]; then
    start_recorder "$name" "$port"
    return
  fi

  if [[ "$SHOW_FPS" == "true" ]]; then
    gst-launch-1.0 -e "${GST_FLAGS[@]}" \
      udpsrc port="$port" caps="application/x-rtp, media=video, encoding-name=H265, payload=${PAYLOAD_TYPE}, clock-rate=90000" \
      ! rtpjitterbuffer latency="$LATENCY_MS" drop-on-latency=true \
      ! rtph265depay \
      ! h265parse \
      ! avdec_h265 \
      ! videoconvert \
      ! fpsdisplaysink name="${name}_fps" text-overlay=false video-sink="$VIDEO_SINK" sync="$SYNC" fps-update-interval="$FPS_INTERVAL_MS" &
  else
    gst-launch-1.0 -e "${GST_FLAGS[@]}" \
      udpsrc port="$port" caps="application/x-rtp, media=video, encoding-name=H265, payload=${PAYLOAD_TYPE}, clock-rate=90000" \
      ! rtpjitterbuffer latency="$LATENCY_MS" drop-on-latency=true \
      ! rtph265depay \
      ! h265parse \
      ! avdec_h265 \
      ! videoconvert \
      ! "$VIDEO_SINK" sync="$SYNC" &
  fi
  PIDS+=("$!")
}

start_recorder() {
  local name="$1"
  local port="$2"
  local ts
  local outfile
  local mux

  ts="$(date +%Y%m%d_%H%M%S)"
  outfile="${RECORD_DIR}/${name}_${ts}.${RECORD_CONTAINER}"
  if [[ "$RECORD_CONTAINER" == "mp4" ]]; then
    mux="mp4mux"
  else
    mux="matroskamux"
  fi

  echo "[receive_h265_dual] recording ${name}: ${outfile}"
  if [[ "$RECORD_PREVIEW" == "true" ]]; then
    gst-launch-1.0 -e "${GST_FLAGS[@]}" \
      udpsrc port="$port" caps="application/x-rtp, media=video, encoding-name=H265, payload=${PAYLOAD_TYPE}, clock-rate=90000" \
      ! rtpjitterbuffer latency="$LATENCY_MS" drop-on-latency=true \
      ! rtph265depay \
      ! h265parse config-interval=-1 \
      ! tee name=t \
      t. ! queue ! "$mux" ! filesink location="$outfile" \
      t. ! queue ! avdec_h265 ! videoconvert ! fpsdisplaysink name="${name}_fps" text-overlay=false video-sink="$VIDEO_SINK" sync="$SYNC" fps-update-interval="$FPS_INTERVAL_MS" &
  else
    gst-launch-1.0 -e "${GST_FLAGS[@]}" \
      udpsrc port="$port" caps="application/x-rtp, media=video, encoding-name=H265, payload=${PAYLOAD_TYPE}, clock-rate=90000" \
      ! rtpjitterbuffer latency="$LATENCY_MS" drop-on-latency=true \
      ! rtph265depay \
      ! h265parse config-interval=-1 \
      ! "$mux" \
      ! filesink location="$outfile" &
  fi
  PIDS+=("$!")
}

if [[ "$RECEIVE_RIGHT" != "true" && "$RECEIVE_LEFT" != "true" ]]; then
  echo "[receive_h265_dual] RECEIVE_RIGHT 和 RECEIVE_LEFT 不能同时为 false"
  exit 1
fi

if [[ "$RECEIVE_RIGHT" == "true" ]]; then
  start_receiver "right" "$RIGHT_PORT"
fi

if [[ "$RECEIVE_LEFT" == "true" ]]; then
  start_receiver "left" "$LEFT_PORT"
fi

if [[ "$RECORD_SECONDS" != "0" ]]; then
  sleep "$RECORD_SECONDS"
  cleanup
  wait 2>/dev/null || true
  print_recordings
  exit 0
fi

wait
print_recordings
