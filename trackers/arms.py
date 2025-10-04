from pose_backend.kinematics import angle, asymmetry_index

class ArmsTracker:
    def __init__(self, cfg):
        self.cfg = cfg; self.reps = 0; self.last_phase = "down"
    def reset(self): self.__init__(self.cfg)

    def update(self, kps):
        L = angle(kps["left_elbow"][:2],  kps["left_shoulder"][:2], kps["left_hip"][:2])
        R = angle(kps["right_elbow"][:2], kps["right_shoulder"][:2], kps["right_hip"][:2])
        asym = asymmetry_index(L, R)
        cue = None
        if asym > self.cfg["metrics"]["asymmetry_warn"]:
            cue = "Try to raise both sides evenly"
        metrics = {"L_shoulder_deg": round(L,1), "R_shoulder_deg": round(R,1), "asymmetry_index": round(asym,2)}
        overlays = {"lines":[["left_shoulder","left_elbow","left_hip"],
                             ["right_shoulder","right_elbow","right_hip"]]}
        return metrics, overlays, cue
