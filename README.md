# UR5e Quest 3 teleop

Hold the right grip to move the arm; the right trigger controls the gripper. For fine motion, release the grip: the stick moves the tool along its own axes (up = +x, right = +y), and A/B move it toward and away from the fingertips along its z axis (`--jog-speed`, 1 cm/s at full stick, A/B at half that). Commands go through the
`ur_robot_driver` already running on `coral`. Do not use `ur_rtde` control alongside it, because it
replaces the driver's program on the arm.

| File | Purpose |
|---|---|
| `core.py` | Kinematics, clutch, and limits (no ROS) |
| `teleop_node.py` | ROS 2 node: Quest input, safety checks, commands |
| `quest_sender.py` | Runs on the Mac: sends Quest poses to `coral` over Tailscale |
| `quest_link.py` | Packet format and the sender allowlist |
| `quest_app.patch` | Our changes to the `oculus_reader` headset app (headset pose, passthrough) |
| `check_quest.py` | Reads the USB-connected Quest for 15 s and prints a summary; sends nothing |
| `record.sh` | Records a demonstration to a rosbag |
| `fake_robot.py` | Driver stand-in for testing on an isolated ROS domain |
| `test_core.py`, `test_quest_link.py` | Unit tests: `.venv/bin/python -m pytest -q` |

## Run

The Quest plugs into the Mac by USB. `quest_sender.py` sends only poses and button states to `coral`
over Tailscale, and `coral` accepts them only from the Mac's Tailscale IP. `coral` gets no access to
the headset.

**On the Mac** (once), install the reader, build the patched app, and install it. The stock APK will
not work: the patched app also streams the headset pose and shows passthrough.

```bash
git lfs install
git clone https://github.com/jborbik/oculus_reader.git third_party/oculus_reader    # gitignored
git -C third_party/oculus_reader apply "$PWD/quest_app.patch"
(cd third_party/oculus_reader/app && JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home" ANDROID_HOME="$HOME/Library/Android/sdk" sh ./gradlew assembleDebug)
.venv/bin/pip install pure-python-adb ./third_party/oculus_reader
adb devices    # the Quest should show as "device"
adb uninstall com.rail.oculus.teleop    # needed once: the signing key differs from the stock APK
adb install third_party/oculus_reader/app/build/outputs/apk/debug/OculusTeleop-debug.apk
.venv/bin/python check_quest.py    # move the right controller, squeeze grip and trigger
```

`quest_app.patch` is the only record of our app changes; the `third_party/oculus_reader` checkout (gitignored) is upstream
code plus this patch, and is not pushed anywhere. If you change the app, regenerate the patch so it
is kept with this folder:

```bash
git -C third_party/oculus_reader diff > quest_app.patch
```

Moving your head does not move the arm. "Forward" is the way the headset faces when you press the
grip, ignoring tilt, and it stays fixed until you release.

Then keep the sender running:

```bash
.venv/bin/python quest_sender.py --to 100.89.37.109
```

**On `coral`**, get the code with git, then start a container on the robot's ROS domain from that
folder:

```bash
git clone https://github.com/MahitMehta/device-assembly-teleop.git ~/device-assembly    # once
cd ~/device-assembly && git pull    # after each push from the Mac
docker run --rm -it --network host \
  -e ROS_DOMAIN_ID=66 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -v "$PWD":/teleop -v /data:/data -w /teleop manip-ros2-bridge:latest bash
source /opt/ros/humble/setup.bash
```

Every `teleop_node.py` command below also needs these two flags:

```bash
--udp-bind 100.89.37.109 --udp-from 100.107.32.64
```

1. **Dry-run (default, sends nothing to the arm or gripper).** Hold grip, push your hand away from
   the headset, and check the logged x/y/z offsets move in the robot's forward direction.
   `--robot-forward-yaw-deg` is that direction in degrees from base +x. It defaults to 180 (base -x),
   measured for this setup; change it only if the arm or workspace moves. It does not change when you
   turn or move around.
   ```bash
   python3 teleop_node.py --robot-forward-yaw-deg 180
   ```
2. **Before going live:**
   - Confirm nobody else is using the arm. Switching controllers interrupts their work.
   - Move the arm to a central pose, away from wrist limits.
   - Set reduced speed limits on the pendant, and have someone at the e-stop.
   - Confirm `speed_scaling` is above 0, with the External Control program running.
   - Activate the controller:
   ```bash
   ros2 service call /controller_manager/switch_controller controller_manager_msgs/srv/SwitchController \
     "{activate_controllers: [forward_position_controller], deactivate_controllers: [scaled_joint_trajectory_controller], strictness: 2, timeout: {sec: 2}}"
   ```
3. **Go live.** The gripper position is the jaw gap in metres (2F-85: 0.085 open, 0 closed).
   ```bash
   python3 teleop_node.py --live --robot-forward-yaw-deg 180 --tcp-z <m> \
     --gripper-open 0.085 --gripper-closed 0
   ```
   Trigger depth sets the jaw gap, limited to `--gripper-speed` (0.04 m/s). The node sends nothing to
   the gripper until the trigger is first pulled, then assumes it starts open.
4. **Record** from a second shell: `./record.sh <task_name>`. Ctrl-C ends the demonstration.
5. **Restore** the previous controller when finished, by reversing the switch in step 2.

## Safety behaviour

The node refuses `--live` unless `forward_position_controller` is active with the expected joint order.
It holds position, and requires a grip release to clear, on any of these:

- robot or safety mode not normal, or speed scaling at 0
- joint states older than 50 ms
- commanded and measured joints more than 0.10 rad apart
- force over 60 N, or more than 30 N above the value when the grip was pressed
- controller pose jump over 10 cm or 0.6 rad between frames

The stick and A/B follow the same holds and limits, and clear by letting go of them. They do nothing while the grip is held, or when tracking or the network drops.

Lost tracking or 250 ms of network silence also holds position, and silence keeps the gripper as it was. When tracking returns it re-engages from the current poses, so the
arm does not jump. Motion is limited to a box of
+/-40 cm around the startup pose (`--box-half-extent`), at 0.10 m/s by default. The box does not know
about the table or the robot's own base: use `--min-z` for a floor. These are software limits: they
do not replace the pendant's safety limits or the e-stop. The log shows `workspace limit` or
`joint limit` when one of these is stopping motion.

## Test without hardware

Use a domain other than 66 so nothing can reach the robot:

```bash
ROS_DOMAIN_ID=199 python3 fake_robot.py --duration 11 &
ROS_DOMAIN_ID=199 python3 teleop_node.py --fake-source --live
```

Add `--fake-jog` to exercise the stick and A/B instead of the grip. To test the network link without a
headset, run `quest_sender.py --fake` on the Mac and start `teleop_node.py` with the `--udp-*` flags
instead of `--fake-source`.

## Still to confirm

- Rosbag-to-LeRobot conversion is not written yet
- Clean-checkout install: clone, patch and APK build verified from scratch on the Mac (2026-11-05); the `pip install` and `adb` steps have not been re-run from scratch
- That `/data` on the host is the right place for demonstrations
