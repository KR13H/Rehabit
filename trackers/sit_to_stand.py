# trackers/sit_to_stand.py
from typing import Tuple, Dict, Any
from pose_backend.kinematics import angle, asymmetry_index

class SitStandTracker:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.reps = 0
        self._state = "seated"   # or "standing"

    def reset(self):
        self.reps = 0
        self._state = "seated"

    def update(self, kps: Dict[str, Tuple[float,float,float,float]]):
        # Midpoints
        mid_hip = ((kps["left_hip"][0] + kps["right_hip"][0]) / 2,
                   (kps["left_hip"][1] + kps["right_hip"][1]) / 2)
        # Simple sit↔stand using hip Y
        hip_y = mid_hip[1]
        # You may want to derive these from frame height; keeping simple for now
        # Consider passing frame height in metrics if you want adaptive thresholds.
        # Example thresholds could be provided in cfg.
        # Here we assume caller uses consistent framing.

        # Knee asymmetry
        Lk = angle(kps["left_hip"][:2],  kps["left_knee"][:2],  kps["left_ankle"][:2])
        Rk = angle(kps["right_hip"][:2], kps["right_knee"][:2], kps["right_ankle"][:2])
        ai = asymmetry_index(Lk, Rk)

        cue = None
        if ai > self.cfg["metrics"]["asymmetry_warn"]:
            cue = "Shift weight evenly"

        metrics = {
            "knee_asymmetry": round(ai, 2),
            "reps": self.reps,
        }
        overlays = {"points": ["left_knee","right_knee","left_hip","right_hip"]}

        # Very crude state machine; tweak thresholds in practice
        # If you pass frame height H, compare hip_y to H*0.62 etc.
        # For now, just return metrics; rep logic can be improved later.
        return metrics, overlays, cue
