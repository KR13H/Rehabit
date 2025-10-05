from dotenv import load_dotenv
load_dotenv()

import os, base64, time, math, traceback, requests
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2, numpy as np, yaml
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from pose_backend.mediapipe_engine import PoseEngine
from pose_backend.schemas import MetricsOut
from pose_backend.utils import valid_keypoints
from trackers.arms import ArmsTracker
from trackers.sit_to_stand import SitStandTracker
from trackers.march import MarchTracker
from services.session_store import SessionStore

# ---------------------- Config ----------------------
ROOT = Path(__file__).parent.resolve()
cfg_path = ROOT / "config" / "settings.yaml"
cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8")) if cfg_path.exists() else {}
cfg.setdefault("model", {}).setdefault("mediapipe", {"detection_conf": 0.5, "tracking_conf": 0.5})
cfg.setdefault("confidence", {}).setdefault("min_visibility", 0.5)
cfg.setdefault("metrics", {}).setdefault("asymmetry_warn", 0.25)
cfg.setdefault("metrics", {}).setdefault("trunk_lean_warn_deg", 30)
cfg.setdefault("server", {}).setdefault("host", "0.0.0.0")
cfg["server"]["port"] = int(os.getenv("PORT", cfg["server"].get("port", 8000)))

# ------------ Flask/SocketIO -------------
app = Flask(
    __name__,
    template_folder=str(ROOT / "web" / "templates"),
    static_folder=str(ROOT / "web" / "static"),
    static_url_path="/static",
)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY") or os.urandom(24).hex()
socketio = SocketIO(app, cors_allowed_origins="*")

def resolve_env_vars(config_services):
    for service in config_services.values():
        for field in ["key", "url"]:
            if field in service and service[field].isupper():
                # Looks up variable from .env/environment
                service[field] = os.getenv(service[field], "")
resolve_env_vars(cfg["services"])

# ------------ Engines/Trackers -------------
pose_engine = PoseEngine(cfg)
trackers = {
    "arms":  ArmsTracker(cfg),
    "sit":   SitStandTracker(cfg),
    "march": MarchTracker(cfg),
}
session_store = SessionStore()

# ----------- (Optional) Protocol container (simplified) ----------
@dataclass
class Step:
    name: str
    mode: str
    target_reps: int
    th: Dict[str, float]
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
    phase: str = "idle"
    cur_peak: Dict[str, float] = field(default_factory=dict)
    records: Dict[str, List[RepRecord]] = field(default_factory=dict)
    started_at: float = 0.0
    last_change_t: float = 0.0
    # ... (skip further internals for brevity) ...

    def start(self) -> Dict[str, Any]:
        self.active = True
        self.step_idx = 0
        self.reps_done = 0
        self.phase = "idle"
        self.cur_peak = {}
        self.records = {s.name: [] for s in self.steps}
        self.started_at = time.time()
        self.last_change_t = self.started_at
        return self.state(first_hint=True)

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
    # ... Further details truncated for brevity ...

PROTOCOL_STEPS: List[Step] = [
    Step("Arm Raise (3 slow reps)",     "arms", 3, th={"shoulder_up": 110.0, "shoulder_down": 70.0},  hint="Raise both arms slowly to head height. 3 reps."),
    Step("Elbow Flexion (3 slow reps)", "arms", 3, th={"elbow_bent": 70.0, "elbow_straight": 150.0},   hint="Bend and straighten both elbows. 3 reps."),
    Step("Sit ↔ Stand (3 reps)",        "sit",  3, th={},                                               hint="Stand fully, then sit down. 3 reps."),
    Step("March in Place (10 steps)",   "march",10, th={},                                              hint="Lift knees alternately until we count 10 steps."),
]
protocol = RepProtocol(PROTOCOL_STEPS)

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

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("start_task")
def start_task(data):
    mode = data.get("mode", "arms")
    trackers[mode].reset()
    emit("task_started", {"ok": True, "mode": mode})

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

        kps = pose_engine.infer(frame)
        if not kps or not valid_keypoints(kps, cfg):
            emit("metrics", MetricsOut.paused("Recenter your body"), json=True); return

        if mode not in trackers:
            emit("metrics", MetricsOut.paused(f"Unknown mode '{mode}'"), json=True); return

        metrics, overlays, cue = trackers[mode].update(kps)
        overlays = {**(overlays or {}), "skeleton": _build_overlay_from_kps(kps)}

        # API cue enrichment for feedback
        tip_text = cue
        audio_b64 = ""
        if cue:
            try:
                # Gemini AI text call
                gem_cfg = cfg.get("services", {}).get("gemini", {})
                if gem_cfg:
                    gem_payload = {
                        "model": gem_cfg["model"],
                        "prompt": f"{cue} Asymmetry index: {metrics.get('asymmetry_index', 0):.2f}. Give one corrective tip in simple words."
                    }
                    gem_headers = {"Authorization": f"Bearer {gem_cfg['key']}"}
                    gem_resp = requests.post(gem_cfg["url"], json=gem_payload, headers=gem_headers, timeout=6)
                    tip_text = gem_resp.json().get("choices", [{}])[0].get("text", cue)
                # ElevenLabs TTS
                eleven_cfg = cfg.get("services", {}).get("elevenlabs", {})
                if eleven_cfg and tip_text:
                    tts_payload = {"text": tip_text}
                    tts_headers = {
                        "xi-api-key": eleven_cfg["key"],
                        "Content-Type": "application/json"
                    }
                    tts_resp = requests.post(eleven_cfg["url"], json=tts_payload, headers=tts_headers, timeout=8)
                    if tts_resp.status_code == 200:
                        audio_b64 = base64.b64encode(tts_resp.content).decode()
            except Exception as e:
                print(f"AI/Voice feedback error: {e}")

        emit("metrics", MetricsOut.ok(metrics, overlays, cue=tip_text), json=True)
        if audio_b64:
            socketio.emit("audio_cue", {"audio": audio_b64})

    except Exception as e:
        traceback.print_exc()
        emit("metrics", MetricsOut.paused(f"{type(e).__name__}: {e}"), json=True)

@socketio.on("finish_session")
def finish_session(data):
    report = session_store.finish()
    emit("session", report, json=True)

# Optional: protocol events for advanced guidance (if used)
@socketio.on("start_protocol")
def start_protocol(_):
    emit("protocol_state", protocol.start())

@socketio.on("stop_protocol")
def stop_protocol(_):
    emit("final_report", protocol.stop())

@socketio.on("frame_ping")
def frame_ping(_):
    emit("pong", {"ok": True})

if __name__ == "__main__":
    socketio.run(app, host=cfg["server"]["host"], port=cfg["server"]["port"])
