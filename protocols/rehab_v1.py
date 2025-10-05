# protocols/rehab_v1.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import time
import math

# Helper
def _deg_ok(v) -> bool:
    return v is not None and not math.isnan(v)

@dataclass
class Step:
    name: str               # "Arm Raise Symmetry"
    mode: str               # "arms" | "sit" | "march"
    target_reps: int
    hints: List[str] = field(default_factory=list)
    # thresholds for rep detection
    th: Dict[str, float] = field(default_factory=dict)

@dataclass
class RepRecord:
    t: float
    metrics: Dict[str, Any]

class RehabProtocolV1:
    """
    Stateless from the outside; stateful per run.
    Call .start(), then on each frame call .update(metrics, mode).
    Emits small dicts you can forward to the UI.
    """
    def __init__(self, cfg: dict):
        # Steps (you can tweak targets)
        self.steps: List[Step] = [
            Step("Calibration", "arms", target_reps=0, hints=["Sit tall. Align yourself in the frame."], th={}),
            Step("Arm Raise Symmetry", "arms", target_reps=6, hints=["Raise both arms slowly to head height."],
                 th={"shoulder_up": 110.0, "shoulder_down": 70.0, "hysteresis": 10.0}),
            Step("Elbow Flexion", "arms", target_reps=6, hints=["Bend and straighten both elbows."],
                 th={"elbow_bent": 70.0, "elbow_straight": 150.0, "hysteresis": 10.0}),
            Step("Sit ↔ Stand", "sit", target_reps=5, hints=["Stand all the way up, then sit down."],
                 th={"hip_rise_px": 60.0, "hysteresis": 10.0}),
            Step("March in Place", "march", target_reps=20, hints=["Lift knees alternately."],
                 th={"knee_up": 60.0, "hysteresis": 10.0}),
        ]
        self.cfg = cfg
        self.reset()

    # -------- lifecycle --------
    def reset(self):
        self.active: bool = False
        self.step_idx: int = 0
        self.reps_done: int = 0
        self.phase: str = "idle"   # "idle"|"up"|"down"
        self.records: Dict[str, List[RepRecord]] = {s.name: [] for s in self.steps}
        self.started_at: float = 0.0
        self.cur_peak: Dict[str, float] = {}  # collect peak angles in current rep

    def start(self) -> Dict[str, Any]:
        self.reset()
        self.active = True
        self.started_at = time.time()
        return self._state()

    def next_step(self) -> Dict[str, Any]:
        if not self.active: return self._state(error="not_active")
        self.reps_done = 0
        self.phase = "idle"
        self.cur_peak = {}
        if self.step_idx < len(self.steps) - 1:
            self.step_idx += 1
        return self._state()

    def stop(self) -> Dict[str, Any]:
        self.active = False
        return {"active": False, "done": True, "report": self._build_report()}

    # -------- frame update (call this every frame) --------
    def update(self, metrics: Dict[str, Any], mode: str) -> Dict[str, Any]:
        if not self.active: 
            return self._state()
        step = self.steps[self.step_idx]
        if step.mode != mode:
            # protocol expects a different tracker; tell UI to switch buttons
            return self._state(hint=f"Switch to {step.mode} mode")
        # Rep logic per step
        if step.name == "Calibration":
            # Collect a few seconds then auto-advance
            if time.time() - self.started_at > 8:
                self.next_step()
            return self._state(hint="Calibrating posture…")
        elif step.name == "Arm Raise Symmetry":
            self._rep_logic_arm_raise(step, metrics)
        elif step.name == "Elbow Flexion":
            self._rep_logic_elbow(step, metrics)
        elif step.name == "Sit ↔ Stand":
            self._rep_logic_sitstand(step, metrics)
        elif step.name == "March in Place":
            self._rep_logic_march(step, metrics)

        # finish step?
        if self.reps_done >= step.target_reps:
            # auto-advance after a brief pause
            if self.phase != "done":
                self.phase = "done"
                self._last_done = time.time()
            elif time.time() - getattr(self, "_last_done", 0) > 1.0:
                self.next_step()

        return self._state()

    # -------- rep detection helpers --------
    def _rep_logic_arm_raise(self, step: Step, m: Dict[str, Any]):
        up_th = step.th["shoulder_up"]; down_th = step.th["shoulder_down"]
        L = m.get("L_shoulder_deg"); R = m.get("R_shoulder_deg")
        if not (_deg_ok(L) and _deg_ok(R)): 
            return
        # peak track
        self.cur_peak["L_shoulder_deg"] = max(self.cur_peak.get("L_shoulder_deg", 0), L)
        self.cur_peak["R_shoulder_deg"] = max(self.cur_peak.get("R_shoulder_deg", 0), R)
        # phase machine
        if self.phase in ("idle","down") and (L>up_th and R>up_th):
            self.phase = "up"
        elif self.phase == "up" and (L<down_th and R<down_th):
            # rep complete
            rec = {
                "max_L_shoulder_deg": round(self.cur_peak.get("L_shoulder_deg", 0),1),
                "max_R_shoulder_deg": round(self.cur_peak.get("R_shoulder_deg", 0),1),
                "shoulder_asymmetry": m.get("shoulder_asymmetry"),
            }
            self.records[step.name].append(RepRecord(time.time(), rec))
            self.reps_done += 1
            self.phase = "down"
            self.cur_peak = {}

    def _rep_logic_elbow(self, step: Step, m: Dict[str, Any]):
        bent = step.th["elbow_bent"]; straight = step.th["elbow_straight"]
        L = m.get("L_elbow_deg"); R = m.get("R_elbow_deg")
        if not (_deg_ok(L) and _deg_ok(R)): 
            return
        self.cur_peak["L_elbow_deg"] = min(self.cur_peak.get("L_elbow_deg", 180), L)  # min angle (most flexion)
        self.cur_peak["R_elbow_deg"] = min(self.cur_peak.get("R_elbow_deg", 180), R)
        if self.phase in ("idle","down") and (L < bent and R < bent):
            self.phase = "up"  # "up" == flexed here
        elif self.phase == "up" and (L > straight and R > straight):
            rec = {
                "min_L_elbow_deg": round(self.cur_peak.get("L_elbow_deg", 180),1),
                "min_R_elbow_deg": round(self.cur_peak.get("R_elbow_deg", 180),1),
                "elbow_asymmetry": m.get("elbow_asymmetry"),
            }
            self.records[step.name].append(RepRecord(time.time(), rec))
            self.reps_done += 1
            self.phase = "down"
            self.cur_peak = {}

    def _rep_logic_sitstand(self, step: Step, m: Dict[str, Any]):
        # Use knee angles if available: stand => knee ~ 170+, sit => knee smaller.
        Lk = m.get("L_knee_deg"); Rk = m.get("R_knee_deg")
        if not (_deg_ok(Lk) and _deg_ok(Rk)):
            return
        # heuristics
        sitting = (Lk < 140 and Rk < 140)
        standing = (Lk > 165 and Rk > 165)
        trunk = m.get("trunk_lean_deg") or 0.0
        if self.phase in ("idle","down") and standing:
            self.phase = "up"
        elif self.phase == "up" and sitting:
            rec = {
                "trunk_lean_avg_deg": round(trunk, 1),
                "stood_fully": True
            }
            self.records[step.name].append(RepRecord(time.time(), rec))
            self.reps_done += 1
            self.phase = "down"

    def _rep_logic_march(self, step: Step, m: Dict[str, Any]):
        # Count a "step" when either knee exceeds threshold
        knee_up = step.th["knee_up"]
        Lk = m.get("L_knee_deg")
        Rk = m.get("R_knee_deg")
        if not (_deg_ok(Lk) and _deg_ok(Rk)):
            return
        # Larger knee flexion angle actually means straighter leg; use a proxy:
        # define "high knee" when hip-knee-ankle angle decreases (flexes) below 140
        left_up = Lk < 140
        right_up = Rk < 140
        if self.phase in ("idle","down") and (left_up or right_up):
            self.phase = "up"
            which = "L" if left_up else "R"
            self.records[step.name].append(RepRecord(time.time(), {"step": which}))
            self.reps_done += 1
        elif self.phase == "up" and (not left_up and not right_up):
            self.phase = "down"

    # -------- reporting --------
    def _build_report(self) -> Dict[str, Any]:
        def summarize(step_name: str):
            recs = self.records.get(step_name, [])
            rows = [r.metrics for r in recs]
            return {
                "count": len(rows),
                "samples": rows[:50],  # keep it small
            }
        # Simple scoring (weights sum to 1.0)
        w = {"Arm Raise Symmetry": 0.35, "Elbow Flexion": 0.25, "Sit ↔ Stand": 0.25, "March in Place": 0.15}
        score = 0.0
        # Shoulder symmetry (lower is better)
        arm_rows = [r.metrics for r in self.records["Arm Raise Symmetry"]]
        if arm_rows:
            asym = [abs(x.get("shoulder_asymmetry") or 0) for x in arm_rows]
            arm_score = max(0.0, 1.0 - (sum(asym)/len(asym)))  # 1.0 perfect symmetry
            score += arm_score * w["Arm Raise Symmetry"]
        elbow_rows = [r.metrics for r in self.records["Elbow Flexion"]]
        if elbow_rows:
            asym = [abs(x.get("elbow_asymmetry") or 0) for x in elbow_rows]
            elb_score = max(0.0, 1.0 - (sum(asym)/len(asym)))
            score += elb_score * w["Elbow Flexion"]
        sit_rows = [r.metrics for r in self.records["Sit ↔ Stand"]]
        if sit_rows:
            lean = [x.get("trunk_lean_avg_deg") or 0 for x in sit_rows]
            lean_penalty = min(1.0, (sum(lean)/len(lean))/45.0)  # >45° = zero
            sit_score = 1.0 - lean_penalty
            score += sit_score * w["Sit ↔ Stand"]
        march_rows = [r.metrics for r in self.records["March in Place"]]
        if march_rows:
            # reward alternation roughly
            seq = "".join([x.get("step","") for x in march_rows])
            alt = sum(1 for i in range(1,len(seq)) if seq[i]!=seq[i-1])
            march_score = alt / max(1, len(seq)-1)
            score += march_score * w["March in Place"]

        score_pct = int(round(score * 100))

        recs = []
        if arm_rows and score_pct < 90:
            recs.append("Practice slow symmetrical arm raises; pause briefly at the top.")
        if elbow_rows and score_pct < 90:
            recs.append("Add light support on the weaker side during elbow bends.")
        if sit_rows and sit_rows and any((x.get('trunk_lean_avg_deg') or 0) > 20 for x in sit_rows):
            recs.append("Keep your chest tall when standing up; scoot forward on the chair first.")
        if not recs:
            recs = ["Great job. Maintain consistency 3–4x/week."]

        return {
            "protocol": "RehabProtocolV1",
            "started_at": self.started_at,
            "duration_s": int(time.time() - self.started_at),
            "per_exercise": {
                s.name: summarize(s.name) for s in self.steps if s.target_reps > 0
            },
            "overall_score": score_pct,
            "recommendations": recs,
        }

    def _state(self, hint: Optional[str] = None, error: Optional[str] = None) -> Dict[str, Any]:
        step = self.steps[self.step_idx]
        return {
            "active": self.active,
            "step_idx": self.step_idx,
            "step_name": step.name,
            "expected_mode": step.mode,
            "target_reps": step.target_reps,
            "reps_done": self.reps_done,
            "phase": self.phase,
            "hint": hint or (step.hints[0] if step.hints else None),
            "error": error,
        }
