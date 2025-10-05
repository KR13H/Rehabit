# trackers/sit_to_stand.py
from __future__ import annotations
from typing import Tuple, Dict, Any, Optional

from pose_backend.kinematics import angle, asymmetry_index

def _ema(prev: Optional[float], new: float, alpha: float = 0.25) -> float:
    return new if prev is None else (alpha * new + (1 - alpha) * prev)

def _ok(v: Optional[float]) -> bool:
    return (v is not None) and (not isinstance(v, float) or v == v)  # NaN check

class SitStandTracker:
    """
    Emits robust, view-invariant sit/stand metrics:

      - nhip: normalized hip height in [0..1] where:
          standing ≈ smaller (hips higher) ~ 0.45–0.62
          sitting  ≈ larger  (hips lower)  ~ 0.74–0.95
      - L_knee_deg / R_knee_deg: knee angles (smoothed), fallback for classification
      - trunk_lean_deg: absolute trunk lean (shoulder->hip vs. vertical)
      - posture: "stand" | "sit" | "mid" (for HUD/debug)

    NOTE: Rep counting is handled in the protocol; this tracker just produces signals.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._ema_Lk: Optional[float] = None
        self._ema_Rk: Optional[float] = None
        self._ema_nhip: Optional[float] = None
        self._last_posture: str = "mid"

        # Visibility threshold: be forgiving so ankles/hips don't drop out
        self._vmin = float(cfg.get("confidence", {}).get("min_visibility", 0.5))
        # Use a lower floor internally for sit/stand anchors
        self._vmin_anchors = min(0.2, self._vmin)

        # Hysteresis thresholds for posture classification from nhip
        self._STAND_MAX = 0.62
        self._SIT_MIN   = 0.74

    def reset(self):
        self._ema_Lk = None
        self._ema_Rk = None
        self._ema_nhip = None
        self._last_posture = "mid"

    # --- helpers ---
    def _pick_anchor(self, kps: Dict[str, Tuple[float, float, float, float]], names) -> Optional[Tuple[float,float]]:
        pts = []
        for nm in names:
            if nm in kps:
                x, y, z, v = kps[nm]
                if v is not None and v >= self._vmin_anchors:
                    pts.append((x, y))
        if not pts:
            return None
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _trunk_lean_deg(self, kps) -> Optional[float]:
        # Use whichever shoulder/hip pair is more visible
        for L, H in (("left_shoulder", "left_hip"), ("right_shoulder", "right_hip")):
            if L in kps and H in kps and kps[L][3] >= self._vmin_anchors and kps[H][3] >= self._vmin_anchors:
                sx, sy = kps[L][0], kps[L][1]
                hx, hy = kps[H][0], kps[H][1]
                # angle between vector (shoulder->hip) and image vertical (0, +1)
                # dx = hx - sx; dy = hy - sy (down is +y). Lean = arctan(|dx|/|dy|)
                dx = hx - sx; dy = hy - sy
                if abs(dy) < 1e-6:
                    return 90.0
                import math
                return abs(math.degrees(math.atan(dx / dy)))
        return None

    # --- main update ---
    def update(self, kps: Dict[str, Tuple[float, float, float, float]]):
        # --- Knee angles (always try to provide these) ---
        Lk = None; Rk = None
        if all(j in kps for j in ("left_hip","left_knee","left_ankle")):
            Lk = angle(kps["left_hip"][:2], kps["left_knee"][:2], kps["left_ankle"][:2])
        if all(j in kps for j in ("right_hip","right_knee","right_ankle")):
            Rk = angle(kps["right_hip"][:2], kps["right_knee"][:2], kps["right_ankle"][:2])

        if _ok(Lk): self._ema_Lk = _ema(self._ema_Lk, float(Lk))
        if _ok(Rk): self._ema_Rk = _ema(self._ema_Rk, float(Rk))

        Lk_s = self._ema_Lk if self._ema_Lk is not None else Lk
        Rk_s = self._ema_Rk if self._ema_Rk is not None else Rk

        # Asymmetry cue
        ai = None
        if _ok(Lk_s) and _ok(Rk_s):
            ai = asymmetry_index(Lk_s, Rk_s)

        # --- Normalized hip height (view-invariant) ---
        sh = self._pick_anchor(kps, ["left_shoulder", "right_shoulder"])
        hip = self._pick_anchor(kps, ["left_hip", "right_hip"])
        ank = self._pick_anchor(kps, ["left_ankle", "right_ankle", "left_heel", "right_heel"])

        nhip = None
        if sh and hip and ank:
            xS, yS = sh; xH, yH = hip; xA, yA = ank
            denom = (yA - yS)
            if abs(denom) > 1e-6:
                nhip_raw = (yH - yS) / denom
                # clamp to a sane range before smoothing
                nhip_raw = max(0.0, min(1.4, nhip_raw))
                self._ema_nhip = _ema(self._ema_nhip, nhip_raw, alpha=0.25)
                nhip = self._ema_nhip

        # --- Posture classification with hysteresis + knee fallback ---
        posture = "mid"
        if nhip is not None:
            if nhip <= self._STAND_MAX:
                posture = "stand"
            elif nhip >= self._SIT_MIN:
                posture = "sit"
            else:
                posture = self._last_posture  # within band: keep previous state (hysteresis)
        else:
            # Fallback purely on knees if anchors not available
            if _ok(Lk_s) and _ok(Rk_s):
                if Lk_s >= 168 and Rk_s >= 168:
                    posture = "stand"
                elif Lk_s <= 145 and Rk_s <= 145:
                    posture = "sit"
                else:
                    posture = "mid"

        self._last_posture = posture

        # --- Trunk lean (optional penalty used by protocol) ---
        trunk = self._trunk_lean_deg(kps)
        if trunk is not None:
            # Cap to a reasonable range for readability
            import math
            trunk = float(max(0.0, min(90.0, trunk)))

        # --- Build outputs ---
        metrics: Dict[str, Any] = {
            # Provide smoothed joints the protocol expects
            "L_knee_deg": float(Lk_s) if _ok(Lk_s) else None,
            "R_knee_deg": float(Rk_s) if _ok(Rk_s) else None,
            "trunk_lean_deg": trunk if trunk is not None else None,
            # Viewpoint-invariant signal for extra robustness/telemetry
            "nhip": float(nhip) if nhip is not None else None,
            "posture": posture,
            # Optional coaching
            "knee_asymmetry": round(ai, 2) if ai is not None else None,
        }

        cue = None
        warn_ai = float(self.cfg.get("metrics", {}).get("asymmetry_warn", 0.25))
        if ai is not None and ai > warn_ai:
            cue = "Shift weight evenly"

        overlays = {
            "points": ["left_knee","right_knee","left_hip","right_hip","left_ankle","right_ankle",
                       "left_shoulder","right_shoulder","left_heel","right_heel"],
            # helpful to visualize what the tracker used
            "debug": {"nhip": metrics["nhip"], "posture": posture}
        }

        return metrics, overlays, cue
