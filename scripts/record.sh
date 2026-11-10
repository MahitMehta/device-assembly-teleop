#!/usr/bin/env bash
# Records one demonstration to a rosbag. Usage: ./scripts/record.sh <task_name>
set -euo pipefail
[ "${ROS_DOMAIN_ID:-}" = "66" ] || { echo "Set ROS_DOMAIN_ID=66 first" >&2; exit 1; }
out="/data/teleop_demos/${1:?task name required}/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$(dirname "$out")"
exec ros2 bag record -o "$out" \
  /joint_states \
  /tcp_pose_broadcaster/pose \
  /force_torque_sensor_broadcaster/wrench \
  /camera/camera/color/image_raw/compressed \
  /camera/camera/color/camera_info \
  /forward_position_controller/commands \
  /teleop/target_pose \
  /teleop/engaged \
  /teleop/status \
  /teleop/gripper_command \
  /teleop/gripper_position \
  /io_and_status_controller/robot_mode \
  /io_and_status_controller/safety_mode \
  /tf_static
