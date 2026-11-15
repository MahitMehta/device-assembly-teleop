"""Camera-mount check: a dark finger-like blob, turned by known angles about a point."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")   # in the robot image (python3-opencv); may be missing on a laptop

from quest_teleop.utils.mount_check import check, finger_pose, mount_offset_deg


def scene(turn_deg=0.0):
    img = np.full((192, 256, 3), 100, np.uint8)                      # the mat
    box = cv2.boxPoints(((100.0, 172.0), (14.0, 44.0), 13.0)).astype(np.int32)   # a slanted dark finger
    cv2.fillPoly(img, [box], (10, 10, 10))
    M = cv2.getRotationMatrix2D((165.0, 110.0), turn_deg, 1.0)
    return cv2.warpAffine(img, M, (256, 192), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def ref():
    p = finger_pose([scene()])
    return {"finger_xy": [p.x, p.y], "finger_angle_deg": p.angle, "finger_area": p.area}


def test_reads_back_the_turn_with_its_sign():
    r = ref()
    for turn in (-4.0, -1.5, 2.0, 3.5):
        p = finger_pose([scene(turn)], near=tuple(r["finger_xy"]), ref_area=r["finger_area"])
        assert abs(mount_offset_deg(p, r) - turn) < 0.5


def test_tolerance_and_not_found():
    r = ref()
    assert check([scene(1.0)], r, 3.0)[0] is True
    assert check([scene(5.0)], r, 3.0)[0] is False
    assert check([np.full((192, 256, 3), 100, np.uint8)], r, 3.0)[0] is None   # no finger in view
