# UR5e Quest 3 teleop

Hold the right grip to move the arm; pull the right trigger to close the gripper and release it to open. For fine motion, release the grip: stick up/down drives the gripper forward/back, level, along the way it faces (tool +y, flattened onto the table); stick left/right turns it about vertical through the TCP (right = clockwise from above); A/B move it toward and away from the fingertips along its z axis. Speeds: `--jog-speed` (1 cm/s at full stick, A/B at half that) and `--jog-yaw-speed` (0.5 rad/s at full stick). Click the stick to toggle **tool-down mode**: the arm turns the gripper to point straight down at the table by itself (at `--max-angular-speed`, in place, keeping its yaw) and keeps it there. While gripping, hand tilt is ignored but twisting your hand still turns the gripper about vertical. It turns only while the controller is tracked and the link is live, and any hold turns the mode off so it never resumes by itself. `--tool-down` starts with it on, so the arm turns down as soon as the node starts. Commands go through the
`ur_robot_driver` already running on `coral`. Do not use `ur_rtde` control alongside it, because it
replaces the driver's program on the arm.

On `coral` everything runs through `compose.yaml` (`docker compose run --rm <service>`); the
Python code runs as `python3 -m quest_teleop...` from this folder, with no build step.

| Path | Purpose |
|---|---|
| `quest_teleop/nodes/teleop_node.py` | ROS 2 node: Quest input, safety checks, arm and gripper commands (runs on `coral`) |
| `quest_teleop/nodes/fake_robot.py` | Driver stand-in for testing on an isolated ROS domain |
| `quest_teleop/utils/core.py` | Kinematics, clutch, and limits (no ROS) |
| `quest_teleop/utils/gripper.py` | Drives the Robotiq 2F-85 directly through the URCap socket on the UR controller (no ROS) |
| `quest_teleop/utils/quest_link.py` | Packet format and the sender allowlist |
| `quest_teleop/tools/quest_sender.py` | Runs on the Mac: sends Quest poses to `coral` over Tailscale |
| `quest_teleop/tools/check_quest.py` | Reads the USB-connected Quest for 15 s and prints a summary; sends nothing |
| `compose.yaml` | `coral` services: `driver` (our UR driver), `teleop`, `record`, `ros` |
| `.example.env` | Settings for compose (IPs, ROS domain): copy to `.env` |
| `Dockerfile` | The image those services run in: `ros:humble-ros-base` plus `ur_robot_driver`, Cyclone DDS and numpy. Built on first use; the code is mounted, so rebuild (`docker compose build`) only when dependencies change |
| `scripts/record.sh` | Records a demonstration to a rosbag |
| `scripts/ros.sh` | Container entrypoint: sources ROS 2, then runs the command |
| `headset/quest_app.patch` | Our changes to the `oculus_reader` headset app (headset pose, passthrough, stick click, highest display refresh rate) |
| `test/` | Unit tests: `.venv/bin/python -m pytest -q` (settings in `pytest.ini`) |

## Run

The Quest plugs into the Mac by USB. `quest_sender` sends only poses and button states to `coral`
over Tailscale, and `coral` accepts them only from the Mac's Tailscale IP. `coral` gets no access to
the headset.

**On the Mac** (once), install the reader, build the patched app, and install it. The stock APK will
not work: the patched app also streams the headset pose and shows passthrough.

```bash
git lfs install
git clone https://github.com/jborbik/oculus_reader.git third_party/oculus_reader    # gitignored
git -C third_party/oculus_reader apply "$PWD/headset/quest_app.patch"
(cd third_party/oculus_reader/app && JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home" ANDROID_HOME="$HOME/Library/Android/sdk" sh ./gradlew assembleDebug)
.venv/bin/pip install pure-python-adb ./third_party/oculus_reader
adb devices    # the Quest should show as "device"
adb uninstall com.rail.oculus.teleop    # needed once: the signing key differs from the stock APK
adb install third_party/oculus_reader/app/build/outputs/apk/debug/OculusTeleop-debug.apk
.venv/bin/python -m quest_teleop.tools.check_quest    # move the right controller, squeeze grip and trigger
```

`headset/quest_app.patch` is the only record of our app changes; the `third_party/oculus_reader` checkout (gitignored) is upstream
code plus this patch, and is not pushed anywhere. If you change the app, regenerate the patch so it
is kept with this folder:

```bash
git -C third_party/oculus_reader diff > headset/quest_app.patch
```

Moving your head does not move the arm. "Forward" is the way the headset faces when you press the
grip, ignoring tilt, and it stays fixed until you release.

Then keep the sender running (`100.89.37.109` is `coral`'s Tailscale IP):

```bash
.venv/bin/python -m quest_teleop.tools.quest_sender --to 100.89.37.109
```

**On `coral`**, everything runs from this folder with Docker Compose, including our own UR driver
(`ur_robot_driver`) on our own ROS domain (67), so nothing depends on other containers in the lab.
Each command runs once in the foreground (Ctrl-C stops it); nothing starts or restarts by itself.

```bash
git clone https://github.com/MahitMehta/device-assembly-teleop.git    # once
cp .example.env .env    # once: IPs and ROS domain; edit SENDER_IP if the headset is on another computer
git pull                # after each push from the Mac
```

1. **Start the driver** The driver runs headless: it sends its own program to the arm, so
   put the pendant in **Remote Control** mode; the pendant's installation is not changed. 
   ```bash
   docker compose run --rm driver
   ```
2. **Dry-run** (second shell; sends nothing to the arm or gripper). Hold grip, push your hand away
   from the headset, and check the logged offsets move in the robot's forward direction.
   `--robot-forward-yaw-deg` is that direction in degrees from base +x. It defaults to 180 (base -x),
   measured for this setup; change it only if the arm or workspace moves.
   ```bash
   docker compose run --rm teleop
   ```
3. **Before going live:** move the arm to a central pose away from wrist limits, set reduced speed
   limits on the pendant, have someone at the e-stop, and make sure the gripper is activated.
   Zero the force sensor with nothing touching the gripper, so the force limits measure from 0:
   ```bash
   docker compose run --rm ros ros2 service call /io_and_status_controller/zero_ftsensor std_srvs/srv/Trigger
   ```
4. **Go live.** Extra arguments go straight to `teleop_node`:
   ```bash
   docker compose run --rm teleop --live
   ```
   Pull the trigger (past halfway) to close the gripper, release it to open. The node drives the
   gripper itself through the Robotiq URCap socket on the UR controller (`--gripper-ip`, port
   63352, gripper `--gripper-sid 3`). `--gripper-speed` and `--gripper-force` are the Robotiq
   registers (0-255). Speed defaults to 50; force defaults to 0, Low Force mode with Secure Grip off. The
   node sends nothing to the gripper until the trigger is first pulled, never activates it
   (activation moves the fingers: do it from the pendant), and `--no-gripper` leaves it alone.
5. **Record** from a third shell: `docker compose run --rm record <task_name>`. Ctrl-C ends the
   demonstration. Only topics on our domain are recorded: a camera has to publish on domain 67
   to be included.
6. **When finished**, Ctrl-C the teleop node and then the driver, so the arm is free for others.

For anything else, `docker compose run --rm ros <command>` runs it with ROS sourced, e.g.
`docker compose run --rm ros ros2 topic list`, or with no command for a shell.

To follow a running service's output from another shell (`docker compose logs` skips `run`
containers), e.g. the teleop node or the driver:

```bash
docker logs -f $(docker ps -q --filter name=teleop-run)
docker logs -f $(docker ps -q --filter name=driver-run)
```

## Safety behaviour

The node refuses `--live` unless `forward_position_controller` is active with the expected joint order.
It holds position, and requires a grip release to clear, on any of these:

- robot or safety mode not normal, or speed scaling at 0
- commanded and measured joints more than 0.10 rad apart
- force over 60 N, or more than 30 N above the value when the grip was pressed
- controller pose jump over 10 cm or 0.6 rad between frames

The stick, A/B and tool-down turning follow the same holds and limits, and clear by letting go of them. They do nothing while the grip is held, or when tracking or the network drops.

Joint states older than 50 ms, lost tracking, or 250 ms of network silence also hold position (without needing a grip release, and tool-down mode stays on), and silence keeps the gripper as it was. When tracking returns it re-engages from the current poses, so the
arm does not jump. Motion is limited to a box of
+/-40 cm around the startup pose (`--box-half-extent`), at 0.20 m/s and 1.0 rad/s by default (`--max-linear-speed`, `--max-angular-speed`). The box does not know
about the table or the robot's own base. `--min-z` is a floor, 1 cm above the arm's base by default (`base` frame z = 0.01): no part of the gripper (a box around it out to the fingertips, `--gripper-length` 0.17 m) and none of the elbow or wrist joints goes below it. The gripper slides along the floor; a step that would take the elbow or wrist below it is refused (`floor limit` in the log). Turning tool-down can lift the arm a little to keep the fingertips clear. These are software limits: they
do not replace the pendant's safety limits or the e-stop. The log shows `workspace limit` or
`joint limit` when one of these is stopping motion.

## Test without hardware

Use a spare domain (199) so nothing can reach the robot; it overrides the one in `.env`:

```bash
ROS_DOMAIN_ID=199 docker compose run --rm ros python3 -m quest_teleop.nodes.fake_robot --duration 11
ROS_DOMAIN_ID=199 docker compose run --rm teleop --fake-source --live --no-gripper   # second shell, within ~1 s
```

Add `--fake-jog` to exercise the stick and A/B instead of the grip. To test the network link without a
headset, run `quest_sender --fake` on the Mac and start `teleop_node` with the `--udp-*` flags
instead of `--fake-source`.

## Still Pending

- Rosbag-to-LeRobot conversion is not written yet