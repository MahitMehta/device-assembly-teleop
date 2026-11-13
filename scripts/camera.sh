#!/usr/bin/env bash
# Wrist camera, on the lab laptop the D435 is plugged into: the RealSense node, plus the hand-eye
# calibration (config/wrist_camera_tf) as a static tool0 -> camera_link transform.
# Extra arguments go to rs_launch.py, e.g. serial_no:=_<serial> when two RealSenses are plugged in.
set -euo pipefail
tf=config/wrist_camera_tf
if [ -f "$tf" ]; then
  # shellcheck disable=SC2046  # the file holds the --x ... --qw flags
  ros2 run tf2_ros static_transform_publisher $(cat "$tf") --frame-id tool0 --child-frame-id camera_link &
else
  echo "No $tf yet (calibrate with quest_teleop.tools.calibrate_wrist_camera): camera not attached to tool0" >&2
fi
exec ros2 launch realsense2_camera rs_launch.py rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30 align_depth.enable:=true "$@"
