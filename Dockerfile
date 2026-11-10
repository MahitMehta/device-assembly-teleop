# Minimal ROS 2 Humble image for teleop on coral: the UR driver and our node. The code is
# mounted at /teleop by compose.yaml, not copied in, so this only needs rebuilding when
# dependencies change.
FROM ros:humble-ros-base
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-ur-robot-driver \
        ros-humble-rmw-cyclonedds-cpp \
        python3-numpy \
    && rm -rf /var/lib/apt/lists/*
