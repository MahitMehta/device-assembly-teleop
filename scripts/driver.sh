#!/usr/bin/env bash
# The UR driver (compose service `driver`): ur_control.launch.py with the given arguments.
# --force-mode also activates force_mode_controller (which the launch loads inactive) as soon as
# the controllers are up, next to forward_position_controller, for policy_node --force-mode on.
# It only makes force mode available: the arm enters it only when the policy node asks.
set -euo pipefail
force=0
args=()
for a in "$@"; do
  if [ "$a" = "--force-mode" ]; then force=1; else args+=("$a"); fi
done

if [ "$force" = 1 ]; then
  # In the background, while the driver starts; prints "force_mode_controller active" or an ERROR.
  python3 "$(dirname "$0")/activate_controller.py" force_mode_controller 2>&1 | sed 's/^/[driver.sh] /' &
fi

# exec: the driver stays the main process, so Ctrl-C stops it as before.
exec ros2 launch ur_robot_driver ur_control.launch.py "${args[@]}"
