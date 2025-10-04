# trackers/arms.py
from __future__ import annotations
from typing import Dict, Tuple, Any, List, Optional
import math

Point = Tuple[float, float, float, float]  # (x, y, z, visibility)

def _angle(a, b, c) -> float:
    """Angle (in degrees) at point b, given 2D points a,b,c."""
    ax, ay = a; bx, by = b; cx, cy = c
    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)
    n1 = math.hypot(*v1); n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cosang = max(-1.0, min(1.0, (v1[0]*v2[0] + v1[1]*v2[1]) / (n1*n2)))
    return math.degrees(math.acos(cosang))

def _deg(val: float) -> Optional[float]:
    return None if (val is None or math.isnan(val)) else round(float(val), 1)

class ArmsTracker:
    """
    Extended 'arms' tracker:
      - Shoulder abduction (L/R)
      - Elbow flexion (L/R)
      - Knee flexion (L/R)  -> appears when legs are in frame
      - Trunk lean (deg from vertical)
      - Asymmetry indices for shoulders & elbows
    """
    def __init__(self, cfg: dict):
        self.vis_thresh = float(cfg.get("confidence", {}).get("min_visibility", 0.5))
        self.asym_warn = float(cfg.get("metrics", {}).get("asymmetry_warn", 0.25))
        self.trunk_warn = float(cfg.get("metrics", {}).get("trunk_lean_warn_deg", 30))
        self._last_cue_t = 0.0

    def reset(self):
        pass

    def _ok(self, p: Point) -> bool:
        return p is not None and p[3] >= self.vis_thresh

    def _get2(self, p: Point) -> Tuple[float, float]:
        return (p[0], p[1])

    def update(self, kps: Dict[str, Point]) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[str]]:
        m: Dict[str, Any] = {}
        overlays: Dict[str, Any] = {}
        cue: Optional[str] = None

        Ls, Rs = kps.get("left_shoulder"),  kps.get("right_shoulder")
        Le, Re = kps.get("left_elbow"),     kps.get("right_elbow")
        Lw, Rw = kps.get("left_wrist"),     kps.get("right_wrist")
        Lh, Rh = kps.get("left_hip"),       kps.get("right_hip")
        Lk, Rk = kps.get("left_knee"),      kps.get("right_knee")
        La, Ra = kps.get("left_ankle"),     kps.get("right_ankle")

        # ---------- Shoulder abduction (angle between upper arm and torso line) ----------
        # Left: angle at shoulder between elbow and hip
        if all(map(self._ok, [Ls, Le, Lh])):
            m["L_shoulder_deg"] = _deg(_angle(self._get2(Le), self._get2(Ls), self._get2(Lh)))
        if all(map(self._ok, [Rs, Re, Rh])):
            m["R_shoulder_deg"] = _deg(_angle(self._get2(Re), self._get2(Rs), self._get2(Rh)))

        # ---------- Elbow flexion (angle at elbow between shoulder and wrist) ----------
        if all(map(self._ok, [Ls, Le, Lw])):
            m["L_elbow_deg"] = _deg(_angle(self._get2(Ls), self._get2(Le), self._get2(Lw)))
        if all(map(self._ok, [Rs, Re, Rw])):
            m["R_elbow_deg"] = _deg(_angle(self._get2(Rs), self._get2(Re), self._get2(Rw)))

        # ---------- Knee flexion (angle at knee between hip and ankle) ----------
        if all(map(self._ok, [Lh, Lk, La])):
            m["L_knee_deg"] = _deg(_angle(self._get2(Lh), self._get2(Lk), self._get2(La)))
        if all(map(self._ok, [Rh, Rk, Ra])):
            m["R_knee_deg"] = _deg(_angle(self._get2(Rh), self._get2(Rk), self._get2(Ra)))

        # ---------- Trunk lean (angle between mid-hip -> mid-shoulder and vertical) ----------
        if all(map(self._ok, [Ls, Rs, Lh, Rh])):
            mid_sh = ((Ls[0] + Rs[0]) / 2.0, (Ls[1] + Rs[1]) / 2.0)
            mid_hp = ((Lh[0] + Rh[0]) / 2.0, (Lh[1] + Rh[1]) / 2.0)
            # angle at mid-hip between a vertical reference point above hip and the shoulder midpoint
            vert_ref = (mid_hp[0], mid_hp[1] - 100.0)
            trunk_deg = _angle(vert_ref, mid_hp, mid_sh)
            # convert 0..180 to deviation from vertical (0 best)
            if trunk_deg is not None and not math.isnan(trunk_deg):
                dev = abs(90.0 - trunk_deg)  # 90 means perfectly vertical line
                m["trunk_lean_deg"] = _deg(dev)

        # ---------- Asymmetry indices ----------
        def asym(a: Optional[float], b: Optional[float]) -> Optional[float]:
            if a is None or b is None:
                return None
            a = float(a); b = float(b)
            denom = max(a, b, 1e-6)
            return round(abs(a - b) / denom, 2)

        if "L_shoulder_deg" in m and "R_shoulder_deg" in m:
            m["shoulder_asymmetry"] = asym(m["L_shoulder_deg"], m["R_shoulder_deg"])
        if "L_elbow_deg" in m and "R_elbow_deg" in m:
            m["elbow_asymmetry"] = asym(m["L_elbow_deg"], m["R_elbow_deg"])

        # ---------- Coaching cue ----------
        # Prefer shoulder asymmetry; fall back to elbow if present
        warn = None
        if m.get("shoulder_asymmetry") is not None and m["shoulder_asymmetry"] >= self.asym_warn:
            warn = "Try to raise both sides evenly"
        elif m.get("elbow_asymmetry") is not None and m["elbow_asymmetry"] >= self.asym_warn:
            warn = "Try to bend/extend both elbows evenly"

        if warn:
            cue = warn

        return m, overlays, cue
