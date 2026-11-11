# Minimal ROS 2 Humble image for teleop on coral: the UR driver and our node. The code is
# mounted at /teleop by compose.yaml, not copied in, so this only needs rebuilding when
# dependencies change.
FROM ros:humble-ros-base
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-ur-robot-driver \
        ros-humble-rmw-cyclonedds-cpp \
        python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# The driver's arm-side script calls set_target_payload with a transition_time on PolyScope
# 5.10+, but our arm (5.13) takes at most 3 arguments, so setting the payload stopped the
# program ("argument_error_at_most:set_target_payload:3:4"). Drop that argument.
RUN f=/opt/ros/humble/share/ur_client_library/resources/external_control.urscript \
    && sed -i 's/set_target_payload(mass, cog, inertia, transition_time)/set_target_payload(mass, cog, inertia)/' "$f" \
    && grep -q 'set_target_payload(mass, cog, inertia)$' "$f"
