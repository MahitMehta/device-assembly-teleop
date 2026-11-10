#!/usr/bin/env bash
# Container entrypoint: source ROS 2, then run the given command.
set -e
source /opt/ros/humble/setup.bash
exec "$@"
