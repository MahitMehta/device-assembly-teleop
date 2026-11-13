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
exec ros2 launch realsense2_camera rs_launch.py rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30 align_depth.enable:=true "${serial[@]}" "$@"
