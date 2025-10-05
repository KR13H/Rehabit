# pose_backend/mediapipe_engine.py
from __future__ import annotations
from typing import Dict, Tuple
import mediapipe as mp
import numpy as np

# (x, y, z, visibility) tuples expected by the rest of the app
Point = Tuple[float, float, float, float]

# MediaPipe pose landmark indices we care about
KP_IDX = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13,    "right_elbow": 14,
    "left_wrist": 15,    "right_wrist": 16,
    "left_hip": 23,      "right_hip": 24,
    "left_knee": 25,     "right_knee": 26,
    "left_ankle": 27,    "right_ankle": 28,
}

class PoseEngine:
    """
    Thin wrapper around MediaPipe Solutions Pose.

    Usage:
        engine = PoseEngine(cfg)
        kps = engine.infer(bgr_frame)  # dict: name -> (x,y,z,visibility)
    """
    def __init__(self, cfg: dict):
        mp_pose = mp.solutions.pose
        mc = int(cfg.get("model", {}).get("mediapipe", {}).get("model_complexity", 1))
        det = float(cfg.get("model", {}).get("mediapipe", {}).get("detection_conf", 0.5))
        trk = float(cfg.get("model", {}).get("mediapipe", {}).get("tracking_conf", 0.5))

        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=mc,              # 0/1/2
            enable_segmentation=False,
            min_detection_confidence=det,
            min_tracking_confidence=trk,
        )

    def infer(self, bgr_frame: np.ndarray) -> Dict[str, Point]:
        """
        Run pose on a BGR image and return a dict of named keypoints.
        Coordinates are in pixel space of the input frame.
        """
        if bgr_frame is None or bgr_frame.size == 0:
            return {}

        h, w = bgr_frame.shape[:2]
        rgb = bgr_frame[:, :, ::-1]  # BGR -> RGB

        res = self._pose.process(rgb)
        if not res.pose_landmarks:
            return {}

        lm = res.pose_landmarks.landmark
        out: Dict[str, Point] = {}
        for name, idx in KP_IDX.items():
            p = lm[idx]
            out[name] = (p.x * w, p.y * h, getattr(p, "z", 0.0), getattr(p, "visibility", 0.0))

        # handy midpoints used by overlays/trackers
        if "left_shoulder" in out and "right_shoulder" in out:
            out["mid_shoulder"] = (
                (out["left_shoulder"][0] + out["right_shoulder"][0]) / 2.0,
                (out["left_shoulder"][1] + out["right_shoulder"][1]) / 2.0,
                0.0,
                1.0,
            )
        if "left_hip" in out and "right_hip" in out:
            out["mid_hip"] = (
                (out["left_hip"][0] + out["right_hip"][0]) / 2.0,
                (out["left_hip"][1] + out["right_hip"][1]) / 2.0,
                0.0,
                1.0,
            )
        return out
