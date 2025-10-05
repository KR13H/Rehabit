# trackers/march.py
from __future__ import annotations
from typing import Dict, Tuple, Any, Optional
import time

from pose_backend.kinematics import angle, asymmetry_index

def _ema(prev: Optional[float], new: float, alpha: float = 0.25) -> float:
    return new if prev is None else (alpha * new + (1 - alpha) * prev)

def _ok(v: Optional[float]) -> bool:
    return (v is not None) and (not isinstance(v, float) or v == v)  # NaN check

class MarchTracker:
    """
    Lightweight marching signal tracker.

    Emits:
      - L_knee_deg, R_knee_deg (smoothed)  -> used by protocol to count steps
      - leg_up: "L" | "R" | None           -> HUD/debug
      - knee_asymmetry (0..1)               -> coaching cue
      - cadence_hz (best-effort)            -> optional UX metric

    Rep counting itself is handled by the protocol, not here.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._ema_Lk: Optional[float] = None
        self._ema_Rk: Optional[float] = None

        # simple cadence est.
        self._last_switch_t: Optional[float] = None
        self._prev_leg_up: Optional[str] = None
        self._cadence_hz: Optional[float] = None

        self._vmin = float(cfg.get("confidence", {}).get("min_visibility", 0.5))
        # be a bit forgiving for legs
        self._vmin_leg = min(0.3, self._vmin)

    def reset(self):
        self._ema_Lk = None
        self._ema_Rk = None
        self._last_switch_t = None
        self._prev_leg_up = None
        self._cadence_hz = None

    def _knee_angle(self, kps, side: str) -> Optional[float]:
        hip = f"{side}_hip"; knee = f"{side}_knee"; ankle = f"{side}_ankle"
        if hip in kps and knee in kps and ankle in kps:
            # Require at least the knee joint to be fairly visible; others a bit lower
            if kps[knee][3] < self._vmin_leg:
                return None
            return angle(kps[hip][:2], kps[knee][:2], kps[ankle][:2])
        return None

    def update(self, kps: Dict[str, Tuple[float, float, float, float]]):
        # --- compute knee angles ---
        Lk = self._knee_angle(kps, "left")
        Rk = self._knee_angle(kps, "right")

        if _ok(Lk): self._ema_Lk = _ema(self._ema_Lk, float(Lk), alpha=0.25)
        if _ok(Rk): self._ema_Rk = _ema(self._ema_Rk, float(Rk), alpha=0.25)

        Lk_s = self._ema_Lk if self._ema_Lk is not None else Lk
        Rk_s = self._ema_Rk if self._ema_Rk is not None else Rk

        # --- detect which leg is "up" (knee lifted -> more flexion -> smaller angle) ---
        leg_up = None
        if _ok(Lk_s) or _ok(Rk_s):
            left_up  = (_ok(Lk_s) and Lk_s < 140.0)
            right_up = (_ok(Rk_s) and Rk_s < 140.0)
            if left_up and not right_up:
                leg_up = "L"
            elif right_up and not left_up:
                leg_up = "R"
            elif left_up and right_up:
                # both up (rare) -> keep previous to avoid flapping
                leg_up = self._prev_leg_up

        # --- crude cadence estimate from alternations ---
        now = time.time()
        if leg_up and self._prev_leg_up and leg_up != self._prev_leg_up:
            if self._last_switch_t is not None:
                dt = now - self._last_switch_t
                if dt > 0.15:  # ignore ultra-fast noise
                    hz = 1.0 / dt
                    # smooth cadence a bit
                    self._cadence_hz = hz if self._cadence_hz is None else (0.3 * hz + 0.7 * self._cadence_hz)
            self._last_switch_t = now
        elif leg_up and self._prev_leg_up is None:
            self._last_switch_t = now
        self._prev_leg_up = leg_up

        # --- asymmetry cue ---
        ai = None
        if _ok(Lk_s) and _ok(Rk_s):
            ai = asymmetry_index(Lk_s, Rk_s)
        cue = None
        warn_ai = float(self.cfg.get("metrics", {}).get("asymmetry_warn", 0.25))
        if ai is not None and ai > warn_ai:
            cue = "Try to lift both knees evenly"

        # --- metrics to feed protocol/HUD ---
        metrics: Dict[str, Any] = {
            "L_knee_deg": float(Lk_s) if _ok(Lk_s) else None,
            "R_knee_deg": float(Rk_s) if _ok(Rk_s) else None,
            "leg_up": leg_up,
            "knee_asymmetry": round(ai, 2) if ai is not None else None,
            "cadence_hz": float(self._cadence_hz) if self._cadence_hz is not None else None,
        }

        overlays = {
            "points": ["left_knee","right_knee","left_hip","right_hip","left_ankle","right_ankle"],
            "debug": {"leg_up": leg_up, "cadence_hz": metrics["cadence_hz"]},
        }

        return metrics, overlays, cue
