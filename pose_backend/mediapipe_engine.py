import mediapipe as mp
import numpy as np

KP_IDX = {
    "nose":0, "left_shoulder":11, "right_shoulder":12, "left_elbow":13, "right_elbow":14,
    "left_wrist":15, "right_wrist":16, "left_hip":23, "right_hip":24, "left_knee":25,
    "right_knee":26, "left_ankle":27, "right_ankle":28
}

class PoseEngine:
    def __init__(self, cfg):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=cfg["model"]["mediapipe"]["detection_conf"],
            min_tracking_confidence=cfg["model"]["mediapipe"]["tracking_conf"]
        )

    def infer(self, bgr_frame):
        rgb = bgr_frame[:, :, ::-1]
        res = self.pose.process(rgb)
        if not res.pose_landmarks: return {}
        lm = res.pose_landmarks.landmark
        h, w = rgb.shape[:2]
        out = {}
        for name, idx in KP_IDX.items():
            p = lm[idx]
            out[name] = (p.x * w, p.y * h, p.z, p.visibility)
        # midpoints you’ll use often
        out["mid_shoulder"] = ((out["left_shoulder"][0] + out["right_shoulder"][0]) / 2,
                               (out["left_shoulder"][1] + out["right_shoulder"][1]) / 2, 0, 1)
        out["mid_hip"] = ((out["left_hip"][0] + out["right_hip"][0]) / 2,
                          (out["left_hip"][1] + out["right_hip"][1]) / 2, 0, 1)
        return out
