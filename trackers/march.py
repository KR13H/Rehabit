from pose_backend.kinematics import angle, asymmetry_index

class MarchTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.steps = 0
        self.last_lift = None

    def reset(self):
        self.steps = 0
        self.last_lift = None

    def update(self, kps):
        left_knee_y = kps["left_knee"][1]
        right_knee_y = kps["right_knee"][1]
        mid_hip_y = kps["mid_hip"][1]

        # Detect step by knee coming above hip line with tolerance
        step_lift = (
            left_knee_y < mid_hip_y - 20
            or right_knee_y < mid_hip_y - 20
        )

        if step_lift and not self.last_lift:
            self.steps += 1
        self.last_lift = step_lift

        # Asymmetry index
        lk = angle(kps["left_hip"][:2], kps["left_knee"][:2], kps["left_ankle"][:2])
        rk = angle(kps["right_hip"][:2], kps["right_knee"][:2], kps["right_ankle"][:2])
        ai = asymmetry_index(lk, rk)

        cue = None
        if ai > self.cfg["metrics"]["asymmetry_warn"]:
            cue = "Try to lift both knees evenly"

        metrics = {
            "steps": self.steps,
            "knee_asymmetry": round(ai, 2)
        }
        overlays = {"points": ["left_knee", "right_knee", "left_hip", "right_hip"]}

        return metrics, overlays, cue
