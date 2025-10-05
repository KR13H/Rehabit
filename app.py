# app.py
from __future__ import annotations

import os, base64, time, math, traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2, numpy as np, yaml
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# ---- local modules you already have ----
from pose_backend.mediapipe_engine import PoseEngine
from pose_backend.schemas import MetricsOut
from pose_backend.utils import valid_keypoints
from trackers.arms import ArmsTracker
from trackers.sit_to_stand import SitStandTracker
from trackers.march import MarchTracker
from services.session_store import SessionStore

# --------------------- Config ---------------------
ROOT = Path(__file__).parent.resolve()
cfg_path = ROOT / "config" / "settings.yaml"
cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8")) if cfg_path.exists() else {}
cfg.setdefault("model", {}).setdefault("mediapipe", {"detection_conf": 0.5, "tracking_conf": 0.5})
cfg.setdefault("confidence", {}).setdefault("min_visibility", 0.5)
cfg.setdefault("metrics", {}).setdefault("asymmetry_warn", 0.25)
cfg.setdefault("metrics", {}).setdefault("trunk_lean_warn_deg", 30)
cfg.setdefault("server", {}).setdefault("host", "0.0.0.0")
cfg["server"]["port"] = int(os.getenv("PORT", cfg["server"].get("port", 8000)))

# --------------------- Flask/SocketIO ---------------------
app = Flask(
    __name__,
    template_folder=str(ROOT / "web" / "templates"),
    static_folder=str(ROOT / "web" / "static"),
    static_url_path="/static",
)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY") or os.urandom(24).hex()
socketio = SocketIO(app, cors_allowed_origins="*")

# --------------------- Engines/Trackers ---------------------
pose_engine = PoseEngine(cfg)
trackers = {
    "arms":  ArmsTracker(cfg),
    "sit":   SitStandTracker(cfg),
    "march": MarchTracker(cfg),
}
session_store = SessionStore(persist_dir=str(ROOT / ".sessions"))

# ===================== REP-BASED Protocol =====================

@dataclass
class Step:
    name: str
    mode: str               # "arms" | "sit" | "march"
    target_reps: int
    th: Dict[str, float]    # thresholds per step
    hint: str

@dataclass
class RepRecord:
    t: float
    metrics: Dict[str, Any]

@dataclass
class RepProtocol:
    steps: List[Step]
    active: bool = False
    step_idx: int = 0
    reps_done: int = 0
    phase: str = "idle"      # "idle" | "up" | "down" | "stand" | "sit" | "done"
    cur_peak: Dict[str, float] = field(default_factory=dict)
    records: Dict[str, List[RepRecord]] = field(default_factory=dict)
    started_at: float = 0.0
    last_change_t: float = 0.0
    rep_cooldown_t: float = 0.0

    # smoothing (for Sit↔Stand)
    ema_Lk: Optional[float] = None
    ema_Rk: Optional[float] = None
    ema_nhip: Optional[float] = None  # normalized hip height (0 at shoulders, 1 at ankles)

    # tempo constraints (seconds)
    min_up_time: float = 0.7
    min_down_time: float = 0.6
    rep_cooldown: float = 0.35

    def start(self) -> Dict[str, Any]:
        self.active = True
        self.step_idx = 0
        self.reps_done = 0
        self.phase = "idle"
        self.cur_peak = {}
        self.records = {s.name: [] for s in self.steps}
        self.started_at = time.time()
        self.last_change_t = self.started_at
        self.rep_cooldown_t = 0.0
        self.ema_Lk = self.ema_Rk = self.ema_nhip = None
        return self.state(first_hint=True)

    def next_step(self) -> Dict[str, Any]:
        self.reps_done = 0
        self.phase = "idle"
        self.cur_peak = {}
        self.last_change_t = time.time()
        self.rep_cooldown_t = 0.0
        self.ema_Lk = self.ema_Rk = self.ema_nhip = None
        if self.step_idx < len(self.steps) - 1:
            self.step_idx += 1
        return self.state(first_hint=True)

    def stop(self) -> Dict[str, Any]:
        self.active = False
        return {"active": False, "done": True, "report": self.report()}

    def finished(self) -> bool:
        return self.active is False

    def state(self, *, first_hint: bool=False, extra_hint: Optional[str]=None) -> Dict[str, Any]:
        s = self.steps[self.step_idx]
        hint = extra_hint or (s.hint if (first_hint or self.phase=="idle") else None)
        return {
            "active": self.active,
            "step_idx": self.step_idx,
            "step_name": s.name,
            "expected_mode": s.mode,
            "reps_done": self.reps_done,
            "target_reps": s.target_reps,
            "phase": self.phase,
            "hint": hint,
        }

    # ------------ Public update ------------
    def update(self, metrics: Dict[str, Any], current_mode: str, kps: Dict[str, Tuple[float,float,float,float]]) -> Dict[str, Any]:
        """Update with latest metrics + raw keypoints.
        Gating: only count reps if the expected mode is active.
        """
        if not self.active:
            return self.state()

        step = self.steps[self.step_idx]
        if current_mode != step.mode:
            return self.state(extra_hint=f"Switching to {step.mode}…")

        now = time.time()
        in_cd = (now - self.rep_cooldown_t) < self.rep_cooldown

        if step.name.startswith("Arm Raise"):
            self._rep_arm_raise(step, metrics, now, in_cd)
        elif step.name.startswith("Elbow Flexion"):
            self._rep_elbow_flex(step, metrics, now, in_cd)
        elif step.name.startswith("Sit ↔ Stand"):
            # NEW: robust, viewpoint-invariant sit/stand using normalized hip height + knee fallback
            self._rep_sit_stand_robust(step, metrics, kps, now, in_cd)
        elif step.name.startswith("March"):
            self._rep_march(step, metrics, now, in_cd)

        # Auto-advance
        if self.reps_done >= step.target_reps and self.phase != "done":
            self.phase = "done"
            return self.state(extra_hint="Great! Moving to the next exercise…")

        if self.phase == "done" and self.reps_done >= step.target_reps:
            if self.step_idx < len(self.steps) - 1:
                return self.next_step()
            else:
                self.stop()
                return {"active": False, "done": True, "report": self.report()}

        return self.state()

    # ------------ Helpers ------------
    @staticmethod
    def _ok(v):
        return (v is not None) and (not (isinstance(v, float) and math.isnan(v)))

    def _ema(self, prev: Optional[float], new: float, alpha: float = 0.25) -> float:
        return new if prev is None else (alpha * new + (1 - alpha) * prev)

    def _enforce_tempo(self, now: float, min_secs: float) -> bool:
        return (now - self.last_change_t) >= min_secs

    # ------------ Rep logic ------------
    def _rep_arm_raise(self, step: Step, m: Dict[str, Any], now: float, in_cd: bool):
        up = step.th["shoulder_up"]; down = step.th["shoulder_down"]
        L = m.get("L_shoulder_deg"); R = m.get("R_shoulder_deg")
        if not self._ok(L) or not self._ok(R): return

        self.cur_peak["L_sh_max"] = max(self.cur_peak.get("L_sh_max", 0.0), float(L))
        self.cur_peak["R_sh_max"] = max(self.cur_peak.get("R_sh_max", 0.0), float(R))

        if (self.phase in ("idle","down")) and (L > up and R > up) and self._enforce_tempo(now, self.min_up_time):
            self.phase = "up"; self.last_change_t = now
        elif (self.phase == "up") and (L < down and R < down) and (not in_cd) and self._enforce_tempo(now, self.min_down_time):
            rec = {
                "max_L_shoulder_deg": round(self.cur_peak.get("L_sh_max", 0), 1),
                "max_R_shoulder_deg": round(self.cur_peak.get("R_sh_max", 0), 1),
                "shoulder_asymmetry": m.get("shoulder_asymmetry"),
            }
            self.records[step.name].append(RepRecord(time.time(), rec))
            self.reps_done += 1
            self.phase = "down"; self.cur_peak = {}
            self.rep_cooldown_t = now; self.last_change_t = now

    def _rep_elbow_flex(self, step: Step, m: Dict[str, Any], now: float, in_cd: bool):
        bent = step.th["elbow_bent"]; straight = step.th["elbow_straight"]
        L = m.get("L_elbow_deg"); R = m.get("R_elbow_deg")
        if not self._ok(L) or not self._ok(R): return

        self.cur_peak["L_elb_min"] = min(self.cur_peak.get("L_elb_min", 180.0), float(L))
        self.cur_peak["R_elb_min"] = min(self.cur_peak.get("R_elb_min", 180.0), float(R))

        if (self.phase in ("idle","down")) and (L < bent and R < bent) and self._enforce_tempo(now, self.min_up_time):
            self.phase = "up"; self.last_change_t = now
        elif (self.phase == "up") and (L > straight and R > straight) and (not in_cd) and self._enforce_tempo(now, self.min_down_time):
            rec = {
                "min_L_elbow_deg": round(self.cur_peak.get("L_elb_min", 180), 1),
                "min_R_elbow_deg": round(self.cur_peak.get("R_elb_min", 180), 1),
                "elbow_asymmetry": m.get("elbow_asymmetry"),
            }
            self.records[step.name].append(RepRecord(time.time(), rec))
            self.reps_done += 1
            self.phase = "down"; self.cur_peak = {}
            self.rep_cooldown_t = now; self.last_change_t = now

    # ---------- NEW robust Sit↔Stand ----------
    def _rep_sit_stand_robust(
        self, step: Step, m: Dict[str, Any],
        kps: Dict[str, Tuple[float,float,float,float]], now: float, in_cd: bool
    ):
        """
        Viewpoint-invariant detection:
        - Primary signal: normalized hip height nhip = (hip_y - shoulder_y) / (ankle_y - shoulder_y).
          Standing -> hips higher (nhip smaller, ~0.45–0.62). Sitting -> hips lower (nhip bigger, ~0.74–0.95).
        - Fallback: knee angles (standing >=168°, sitting <=140°).
        - Uses EMA smoothing + hysteresis + hold times.
        Works even if user is side-on or chair is off-center.
        """
        def pick(names):
            pts = []
            for nm in names:
                if nm in kps:
                    x,y,z,v = kps[nm]
                    if v is None or (isinstance(v,float) and math.isnan(v)) or v < 0.35:
                        continue
                    pts.append((x,y))
            if not pts: return None
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            return (sum(xs)/len(xs), sum(ys)/len(ys))

        # Key anchors (use any visible side)
        sh = pick(["left_shoulder","right_shoulder"])
        hip = pick(["left_hip","right_hip"])
        ank = pick(["left_ankle","right_ankle","left_heel","right_heel"])
        if not (sh and hip and ank):
            # fallback entirely to knee angles if hips/shoulders/ankles not all visible
            Lk = m.get("L_knee_deg"); Rk = m.get("R_knee_deg")
            if not self._ok(Lk) or not self._ok(Rk): return
            standing = (Lk >= 168 and Rk >= 168)
            sitting  = (Lk <= 140 and Rk <= 140)
        else:
            xS,yS = sh; xH,yH = hip; xA,yA = ank
            # protect against degenerate division
            denom = (yA - yS)
            if abs(denom) < 1e-3: return
            nhip_raw = (yH - yS) / denom  # normalized 0..1 from shoulders->ankles (image y increases downward)
            self.ema_nhip = self._ema(self.ema_nhip, float(nhip_raw), alpha=0.25)
            nhip = self.ema_nhip

            # knee smoothing too (optional)
            Lk = m.get("L_knee_deg"); Rk = m.get("R_knee_deg")
            if self._ok(Lk): self.ema_Lk = self._ema(self.ema_Lk, float(Lk))
            if self._ok(Rk): self.ema_Rk = self._ema(self.ema_Rk, float(Rk))
            Lk_s = self.ema_Lk if self.ema_Lk is not None else Lk
            Rk_s = self.ema_Rk if self.ema_Rk is not None else Rk

            # Hysteresis thresholds (tuneable)
            STAND_MAX = 0.62   # nhip <= this => standing
            SIT_MIN   = 0.74   # nhip >= this => sitting

            standing_sig = (nhip is not None) and (nhip <= STAND_MAX)
            sitting_sig  = (nhip is not None) and (nhip >= SIT_MIN)

            # Knee confirmation (if available) to increase robustness
            knee_stand = (self._ok(Lk_s) and self._ok(Rk_s) and Lk_s >= 165 and Rk_s >= 165)
            knee_sit   = (self._ok(Lk_s) and self._ok(Rk_s) and Lk_s <= 145 and Rk_s <= 145)

            # Final state using OR with knee confirmation (any one strong signal is enough)
            standing = bool(standing_sig or knee_stand)
            sitting  = bool(sitting_sig  or knee_sit)

        # Phase machine: sit -> stand -> sit
        if self.phase in ("idle","sit","down"):
            if sitting:
                self.phase = "sit"
            if standing and self._enforce_tempo(now, self.min_up_time):
                self.phase = "stand"; self.last_change_t = now

        elif self.phase == "stand":
            # require proper return to sit with tempo + cooldown
            if sitting and (not in_cd) and self._enforce_tempo(now, self.min_down_time):
                rec = {
                    "trunk_lean_avg_deg": float(m.get("trunk_lean_deg") or 0.0),
                    "stood_fully": True
                }
                self.records[step.name].append(RepRecord(time.time(), rec))
                self.reps_done += 1
                self.phase = "sit"
                self.rep_cooldown_t = now
                self.last_change_t = now

    def _rep_march(self, step: Step, m: Dict[str, Any], now: float, in_cd: bool):
        Lk = m.get("L_knee_deg"); Rk = m.get("R_knee_deg")
        if not self._ok(Lk) or not self._ok(Rk): return
        left_up  = (Lk < 140)
        right_up = (Rk < 140)
        if (self.phase in ("idle","down")) and (left_up or right_up) and self._enforce_tempo(now, 0.2):
            self.phase = "up"; self.last_change_t = now
            self._which_up = "L" if left_up else "R"
        elif (self.phase == "up") and (not left_up and not right_up) and (not in_cd) and self._enforce_tempo(now, 0.2):
            self.records[step.name].append(RepRecord(time.time(), {"step": self._which_up}))
            self.reps_done += 1
            self.phase = "down"
            self.rep_cooldown_t = now; self.last_change_t = now

    # ------------ Report ------------
    def report(self) -> Dict[str, Any]:
        def avg(arr):
            arr = [float(x) for x in arr if isinstance(x,(int,float))]
            return float(sum(arr)/max(1,len(arr)))
        per_step = {}
        score = 0.0
        weights = {
            "Arm Raise (3 slow reps)": 0.4,
            "Elbow Flexion (3 slow reps)": 0.2,
            "Sit ↔ Stand (3 reps)": 0.25,
            "March in Place (10 steps)": 0.15
        }

        arm = [r.metrics for r in self.records.get("Arm Raise (3 slow reps)", [])]
        if arm:
            asym = [abs(x.get("shoulder_asymmetry") or 0.0) for x in arm]
            s = max(0.0, 1.0 - avg(asym)); score += s*weights["Arm Raise (3 slow reps)"]
            per_step["Arm Raise (3 slow reps)"] = {"avg_asymmetry": round(avg(asym),2), "score_0_1": round(s,2), "reps": arm}

        elb = [r.metrics for r in self.records.get("Elbow Flexion (3 slow reps)", [])]
        if elb:
            asym = [abs(x.get("elbow_asymmetry") or 0.0) for x in elb]
            s = max(0.0, 1.0 - avg(asym)); score += s*weights["Elbow Flexion (3 slow reps)"]
            per_step["Elbow Flexion (3 slow reps)"] = {"avg_asymmetry": round(avg(asym),2), "score_0_1": round(s,2), "reps": elb}

        sit = [r.metrics for r in self.records.get("Sit ↔ Stand (3 reps)", [])]
        if sit:
            trunk = [x.get("trunk_lean_avg_deg") or 0.0 for x in sit]
            penalty = min(1.0, (avg(trunk))/45.0)
            s = 1.0 - penalty; score += s*weights["Sit ↔ Stand (3 reps)"]
            per_step["Sit ↔ Stand (3 reps)"] = {"avg_trunk_lean_deg": round(avg(trunk),1), "score_0_1": round(s,2), "reps": sit}

        mch = [r.metrics for r in self.records.get("March in Place (10 steps)", [])]
        if mch:
            seq = "".join([x.get("step","") for x in mch])
            alt = sum(1 for i in range(1,len(seq)) if seq[i]!=seq[i-1])
            s = (alt / max(1,len(seq)-1)); score += s*weights["March in Place (10 steps)"]
            per_step["March in Place (10 steps)"] = {"alternation_score_0_1": round(s,2), "reps": mch}

        score_pct = int(round(max(0.0, min(1.0, score))*100))
        recs = []
        if per_step.get("Arm Raise (3 slow reps)",{}).get("avg_asymmetry",0)>0.3: recs.append("Practice slow symmetrical arm raises; pause at the top.")
        if per_step.get("Sit ↔ Stand (3 reps)",{}).get("avg_trunk_lean_deg",0)>20: recs.append("Keep chest tall while standing; slide feet back first.")
        if not recs: recs=["Great work! Repeat 4–5 days/week and track improvements."]

        return {
            "protocol": "Rehabit Rep-Guide",
            "started_at": self.started_at,
            "overall_score": score_pct,
            "per_exercise": per_step,
            "recommendations": recs,
        }

# Steps & thresholds (unchanged)
PROTOCOL_STEPS: List[Step] = [
    Step("Arm Raise (3 slow reps)",     "arms", 3, th={"shoulder_up": 110.0, "shoulder_down": 70.0},  hint="Raise both arms slowly to head height. 3 reps."),
    Step("Elbow Flexion (3 slow reps)", "arms", 3, th={"elbow_bent": 70.0, "elbow_straight": 150.0},   hint="Bend and straighten both elbows. 3 reps."),
    Step("Sit ↔ Stand (3 reps)",        "sit",  3, th={},                                               hint="Stand fully, then sit down. 3 reps."),
    Step("March in Place (10 steps)",   "march",10, th={},                                              hint="Lift knees alternately until we count 10 steps."),
]
protocol = RepProtocol(PROTOCOL_STEPS)

# --------------------- Overlays ---------------------
def _build_overlay_from_kps(kps: dict) -> dict:
    pts, names = [], [
        "left_shoulder","right_shoulder","left_elbow","right_elbow",
        "left_wrist","right_wrist","left_hip","right_hip",
        "left_knee","right_knee","left_ankle","right_ankle","left_heel","right_heel"
    ]
    for n in names:
        if n in kps:
            x,y,_,v = kps[n]
            pts.append({"name": n, "x": float(x), "y": float(y), "v": float(v)})
    lines = [
        ["left_shoulder","left_elbow"],["left_elbow","left_wrist"],
        ["right_shoulder","right_elbow"],["right_elbow","right_wrist"],
        ["left_shoulder","right_shoulder"],
        ["left_shoulder","left_hip"],["right_shoulder","right_hip"],
        ["left_hip","right_hip"],
        ["left_hip","left_knee"],["left_knee","left_ankle"],
        ["right_hip","right_knee"],["right_knee","right_ankle"],
    ]
    return {"kp": pts, "lines": lines}

# --------------------- Routes ---------------------
@app.route("/")
def index():
    return render_template("index.html")

# --------------------- Socket events ---------------------
@socketio.on("start_protocol")
def start_protocol(_):
    emit("protocol_state", protocol.start())

@socketio.on("stop_protocol")
def stop_protocol(_):
    emit("final_report", protocol.stop())

@socketio.on("frame")
def on_frame(data):
    try:
        mode = (data or {}).get("mode", "arms")
        img_b64 = (data or {}).get("img_b64", "")
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]

        img = base64.b64decode(img_b64)
        frame = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            emit("metrics", MetricsOut.paused("Bad image payload"), json=True); return

        kps = pose_engine.infer(frame)  # dict: {name: (x,y,z,v)}
        if not kps or not valid_keypoints(kps, cfg):
            emit("metrics", MetricsOut.paused("Recenter your body"), json=True); return

        if mode not in trackers:
            emit("metrics", MetricsOut.paused(f"Unknown mode '{mode}'"), json=True); return

        metrics, overlays, cue = trackers[mode].update(kps)
        overlays = {**(overlays or {}), "skeleton": _build_overlay_from_kps(kps)}

        # >>> Pass BOTH metrics and raw keypoints for robust sit/stand <<<
        state = protocol.update(metrics, mode, kps)
        emit("protocol_state", {
            **state,
            "reps_left": max(0, state.get("target_reps",0) - state.get("reps_done",0))
        })

        emit("metrics", MetricsOut.ok(metrics, overlays, cue), json=True)

        if state.get("done"):
            emit("final_report", state)

    except Exception as e:
        traceback.print_exc()
        emit("metrics", MetricsOut.paused(f"{type(e).__name__}: {e}"), json=True)

@socketio.on("frame_ping")
def frame_ping(_):
    emit("pong", {"ok": True})

# --------------------- Main ---------------------
if __name__ == "__main__":
    socketio.run(app, host=cfg["server"]["host"], port=cfg["server"]["port"])
