#!/usr/bin/env bash
# Wrist camera, on the lab laptop the D435 is plugged into: the RealSense node, plus the hand-eye
# calibration (config/wrist_camera_tf) as a static tool0 -> camera_link transform.
# CAMERA_SERIAL (.env) picks the wrist camera when more than one RealSense is plugged in; without it
# the node opens whichever it finds first. Extra arguments go to rs_launch.py.
set -euo pipefail
tf=config/wrist_camera_tf
if [ -f "$tf" ]; then
  # shellcheck disable=SC2046  # the file holds the --x ... --qw flags
  ros2 run tf2_ros static_transform_publisher $(cat "$tf") --frame-id tool0 --child-frame-id camera_link &
else
  echo "No $tf yet (calibrate with quest_teleop.tools.calibrate_wrist_camera): camera not attached to tool0" >&2
fi
serial=()
# The leading _ keeps ROS from reading an all-digit serial as a number.
[ -n "${CAMERA_SERIAL:-}" ] && serial=("serial_no:=_${CAMERA_SERIAL}")

# Colour exposure, gain and white balance from .env (empty = the camera's automatic setting).
# rs_launch.py takes these only from a config file; it warns that they are "not supported"
# but passes them to the node.
settings=/tmp/camera_settings.yaml
: > "$settings"
number() { [[ "$2" =~ ^[0-9]+$ ]] || { echo "$1 must be a whole number, got '$2'" >&2; exit 1; }; }
if [ -n "${CAMERA_EXPOSURE:-}" ]; then
  number CAMERA_EXPOSURE "$CAMERA_EXPOSURE"
  printf 'rgb_camera.enable_auto_exposure: false\nrgb_camera.exposure: %d\n' "$CAMERA_EXPOSURE" >> "$settings"
fi
if [ -n "${CAMERA_GAIN:-}" ]; then
  number CAMERA_GAIN "$CAMERA_GAIN"
  printf 'rgb_camera.gain: %d\n' "$CAMERA_GAIN" >> "$settings"
fi
if [ -n "${CAMERA_WHITE_BALANCE:-}" ]; then
  number CAMERA_WHITE_BALANCE "$CAMERA_WHITE_BALANCE"
  # A float: the camera steps white balance by 10 K, so the node declares it as a double.
  printf 'rgb_camera.enable_auto_white_balance: false\nrgb_camera.white_balance: %d.0\n' "$CAMERA_WHITE_BALANCE" >> "$settings"
fi
config=()
if [ -s "$settings" ]; then
  echo "Colour camera settings:" >&2; cat "$settings" >&2
  config=("config_file:=$settings")
fi
exec ros2 launch realsense2_camera rs_launch.py rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30 align_depth.enable:=true "${serial[@]}" "${config[@]}" "$@"
