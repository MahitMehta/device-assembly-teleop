#!/usr/bin/env bash
# Records one demonstration to a rosbag. Usage: ./scripts/record.sh <task_name>
# Enter (or Ctrl-C) ends the demonstration; then y keeps it and n deletes it.
set -euo pipefail
[ -n "${ROS_DOMAIN_ID:-}" ] || { echo "Set ROS_DOMAIN_ID first (compose.yaml uses 67, our driver's domain)" >&2; exit 1; }
[ -t 0 ] || { echo "record needs a terminal for its prompts: run it without -T" >&2; exit 1; }
out="/data/teleop_demos/${1:?task name required}/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$(dirname "$out")"

# Job control puts the recorder in its own process group, so Ctrl-C reaches only this script,
# which then stops the recorder cleanly instead of both racing to exit.
set -m
ros2 bag record -o "$out" \
  /joint_states \
  /tcp_pose_broadcaster/pose \
  /force_torque_sensor_broadcaster/wrench \
  /camera/camera/color/image_raw/compressed \
  /camera/camera/color/camera_info \
  /camera/camera/aligned_depth_to_color/image_raw/compressedDepth \
  /camera/camera/aligned_depth_to_color/camera_info \
  /forward_position_controller/commands \
  /teleop/target_pose \
  /teleop/engaged \
  /teleop/status \
  /teleop/gripper_command \
  /teleop/gripper_position \
  /io_and_status_controller/robot_mode \
  /io_and_status_controller/safety_mode \
  /tf_static < /dev/null > /tmp/record.log 2>&1 &  # no terminal: a background job touching it gets stopped
recorder=$!
set +m

sleep 1
kill -0 "$recorder" 2>/dev/null || { cat /tmp/record.log >&2; echo "Recorder failed to start" >&2; exit 1; }
echo "Recording to $out"
# Enter or Ctrl-C ends it. A trapped Ctrl-C does not interrupt bash's read, so read in 1 s slices.
stop=0
trap 'stop=1' INT
printf "Press Enter when the demonstration is done... "
while [ "$stop" = 0 ]; do
  read -r -t 1 _ && break
  [ $? -gt 128 ] || break  # >128 is the 1 s timeout; anything else (end of input) also ends it
done
echo
trap - INT
kill -INT "$recorder" 2>/dev/null || true
kill -CONT "$recorder" 2>/dev/null || true  # in case it was stopped, so it can act on the INT
wait "$recorder" || true

if [ ! -f "$out/metadata.yaml" ]; then
  echo "Recorder did not finish writing $out:" >&2
  tail -5 /tmp/record.log >&2
  exit 1
fi

# Enough to tell a good take from a broken one: length, and whether the camera and arm were heard.
info=$(ros2 bag info "$out" 2>/dev/null || true)
duration=$(sed -n 's/^Duration: *//p' <<<"$info")
count() { grep -F "Topic: $1 |" <<<"$info" | sed -n 's/.*Count: \([0-9]*\).*/\1/p' | head -1 || true; }  # missing topic: empty, not an exit
colour=$(count /camera/camera/color/image_raw/compressed); depth=$(count /camera/camera/aligned_depth_to_color/image_raw/compressedDepth)
joints=$(count /joint_states)
echo "Duration ${duration:-?}: ${colour:-0} colour frames, ${depth:-0} depth frames, ${joints:-0} joint states"
[ "${colour:-0}" -gt 0 ] && [ "${depth:-0}" -gt 0 ] || echo "WARNING: no camera images in this recording (is the camera service running?)"
[ "${joints:-0}" -gt 0 ] || echo "WARNING: no joint states in this recording (is the driver running?)"

while true; do
  read -r -p "Keep this recording? [y/n] " answer || answer=y  # no answer (terminal closed): keep it
  case "$answer" in
    [yY]*) echo "Kept $out"; break ;;
    [nN]*) rm -rf "$out"; echo "Deleted $out"; break ;;
  esac
done
