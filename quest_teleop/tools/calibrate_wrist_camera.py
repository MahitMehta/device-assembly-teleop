"""Hand-eye calibration of the wrist camera (eye-in-hand) with the lab's ArUco grid board.

Fix the board flat to the table, start the driver and the camera, then run this on the camera laptop.
For 15-20 poses: move the arm so the board is in view, let it settle, press Enter to take a sample.
Rotate the wrist a lot between samples (30 deg or more, about different axes); translations alone
give a confident wrong answer. Type `s` + Enter to solve: it writes config/wrist_camera_tf, which
the camera service publishes as tool0 -> camera_link.
"""
import argparse
import os
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage
from tf2_ros import Buffer, TransformListener

# Albus's board (ROS2-essentials, cells/corallab-Dell-G15-5520.env): 3x4 markers from DICT_5X5_50,
# ids from 0, 57.5 mm squares (measured), 6 mm gaps.
DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
BOARD = cv2.aruco.GridBoard_create(3, 4, 0.0575, 0.006, DICT, 0)
CAMERA = "camera/camera/color"
OPTICAL, ROOT = "camera_color_optical_frame", "camera_link"
OUT = "config/wrist_camera_tf"


def to_matrix(t):
    q = t.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    m = np.eye(4)
    m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    m[:3, 3] = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
    return m


def quaternion(r):
    w = np.sqrt(max(0.0, 1 + np.trace(r))) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    return x, y, z, w


def solve(base_tool, cam_board):
    """tool0 -> camera optical frame from (base->tool0, optical->board) pairs, plus the spread of the
    board's position in the base frame across samples (m), which should be a few mm at most."""
    r, t = cv2.calibrateHandEye([m[:3, :3] for m in base_tool], [m[:3, 3] for m in base_tool],
                                [m[:3, :3] for m in cam_board], [m[:3, 3] for m in cam_board],
                                method=cv2.CALIB_HAND_EYE_PARK)
    x = np.eye(4)
    x[:3, :3], x[:3, 3] = r, t.ravel()
    board = np.array([(a @ x @ b)[:3, 3] for a, b in zip(base_tool, cam_board)])
    return x, np.linalg.norm(board - board.mean(axis=0), axis=1).max()


class Calibrator(Node):
    def __init__(self, base):
        super().__init__("calibrate_wrist_camera")
        self.base, self.image, self.info = base, None, None
        self.samples = []
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.create_subscription(CompressedImage, f"/{CAMERA}/image_raw/compressed",
                                 lambda m: setattr(self, "image", m), 1)
        self.create_subscription(CameraInfo, f"/{CAMERA}/camera_info",
                                 lambda m: setattr(self, "info", m), 1)

    def sample(self):
        if self.image is None or self.info is None:
            return print("no image or camera_info yet")
        gray = cv2.imdecode(np.frombuffer(self.image.data, np.uint8), cv2.IMREAD_GRAYSCALE)
        k, d = np.reshape(self.info.k, (3, 3)), np.array(self.info.d)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, DICT)
        n, rvec, tvec = (cv2.aruco.estimatePoseBoard(corners, ids, BOARD, k, d, None, None)
                         if ids is not None else (0, None, None))
        if n < 4:
            return print(f"only {n} board markers seen, need 4+: sample not taken")
        cam_board = np.eye(4)
        cam_board[:3, :3], cam_board[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
        base_tool = to_matrix(self.tf.lookup_transform(self.base, "tool0", Time()))
        self.samples.append((base_tool, cam_board))
        print(f"sample {len(self.samples)}: {n}/12 markers, board {np.linalg.norm(tvec):.2f} m away")

    def write(self):
        if len(self.samples) < 5:
            return print("need at least 5 samples (15-20 is better)")
        tool_optical, spread = solve(*zip(*self.samples))
        root_optical = to_matrix(self.tf.lookup_transform(ROOT, OPTICAL, Time()))
        tool_root = tool_optical @ np.linalg.inv(root_optical)
        x, y, z = tool_root[:3, 3]
        qx, qy, qz, qw = quaternion(tool_root[:3, :3])
        args = f"--x {x:.5f} --y {y:.5f} --z {z:.5f} --qx {qx:.6f} --qy {qy:.6f} --qz {qz:.6f} --qw {qw:.6f}"
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as f:
            f.write(args + "\n")
        print(f"tool0 -> {ROOT}: {args}\n|t| = {np.linalg.norm(tool_root[:3, 3]) * 100:.1f} cm "
              f"(check against the mount); board position spread {spread * 1000:.1f} mm "
              f"over {len(self.samples)} samples (want a few mm). Wrote {OUT}; restart the camera service.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-frame", default="base_link")
    a = p.parse_args()
    rclpy.init()
    node = Calibrator(a.base_frame)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    print("Enter: take a sample   s + Enter: solve and write   q + Enter: quit")
    while (line := input().strip().lower()) != "q":
        try:
            node.write() if line == "s" else node.sample()
        except Exception as e:  # e.g. TF not available yet; keep the samples taken so far
            print(f"error: {e}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
