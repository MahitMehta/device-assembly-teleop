#!/usr/bin/env bash
# Usage: switch_controller.sh teleop|restore
# teleop:  forward_position_controller on (what teleop_node commands)
# restore: back to scaled_joint_trajectory_controller
# Check nobody else is using the arm first: switching interrupts their work.
set -euo pipefail
case "${1:-}" in
  teleop)  on=forward_position_controller;        off=scaled_joint_trajectory_controller ;;
  restore) on=scaled_joint_trajectory_controller; off=forward_position_controller ;;
  *) echo "Usage: $0 teleop|restore" >&2; exit 1 ;;
esac
exec ros2 service call /controller_manager/switch_controller controller_manager_msgs/srv/SwitchController \
  "{activate_controllers: [$on], deactivate_controllers: [$off], strictness: 2, timeout: {sec: 2}}"
