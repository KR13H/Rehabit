# app.py
from __future__ import annotations

import os, base64, traceback
from pathlib import Path

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

ROOT = Path(__file__).parent.resolve()
cfg_path = ROOT / "config" / "settings.yaml"
cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8")) if cfg_path.exists() else {}
cfg.setdefault("model", {}).setdefault("mediapipe", {"detection_conf": 0.5, "tracking_conf": 0.5})
cfg.setdefault("confidence", {}).setdefault("min_visibility", 0.5)
cfg.setdefault("metrics", {}).setdefault("asymmetry_warn", 0.25)
cfg.setdefault("metrics", {}).setdefault("trunk_lean_warn_deg", 30)
cfg.setdefault("server", {}).setdefault("host", "0.0.0.0")
cfg["server"]["port"] = int(os.getenv("PORT", cfg["server"].get("port", 5001)))

app = Flask(
    __name__,
    template_folder=str(ROOT / "web" / "templates"),
    static_folder=str(ROOT / "web" / "static"),
    static_url_path="/static",
)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY") or os.urandom(24).hex()
socketio = SocketIO(app, cors_allowed_origins="*")

pose_engine = PoseEngine(cfg)
trackers = {"arms": ArmsTracker(cfg), "sit": SitStandTracker(cfg), "march": MarchTracker(cfg)}
session_store = SessionStore(persist_dir=str(ROOT / ".sessions"))

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("start_task")
def start_task(data):
    mode = (data or {}).get("mode", "arms")
    if mode not in trackers:
        emit("task_started", {"ok": False, "error": f"unknown mode '{mode}'"})
        return
    trackers[mode].reset()
    emit("task_started", {"ok": True, "mode": mode})

def _build_overlay_from_kps(kps: dict) -> dict:
    """Create a minimal set of points/lines for client-side drawing."""
    pts = []
    names = [
        "left_shoulder","right_shoulder","left_elbow","right_elbow",
        "left_wrist","right_wrist","left_hip","right_hip",
        "left_knee","right_knee","left_ankle","right_ankle"
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

@socketio.on("frame")
def on_frame(data):
    try:
        mode = (data or {}).get("mode", "arms")
        if mode not in trackers:
            emit("metrics", MetricsOut.paused(f"Unknown mode '{mode}'"), json=True); return

        img_b64 = (data or {}).get("img_b64", "")
        if not img_b64:
            emit("metrics", MetricsOut.paused("No frame provided"), json=True); return
        if "," in img_b64:  # handle data URL prefix
            img_b64 = img_b64.split(",", 1)[1]

        img = base64.b64decode(img_b64)
        frame = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            emit("metrics", MetricsOut.paused("Bad image payload"), json=True); return

        kps = pose_engine.infer(frame)  # {name: (x,y,z,vis)}
        if not kps or not valid_keypoints(kps, cfg):
            emit("metrics", MetricsOut.paused("Recenter your body"), json=True); return

        metrics, overlays, cue = trackers[mode].update(kps)

        # add a simple skeleton overlay (client will draw it)
        skel = _build_overlay_from_kps(kps)
        if overlays:
            # merge with tracker-provided overlays if any
            overlays = {**overlays, "skeleton": skel}
        else:
            overlays = {"skeleton": skel}

        emit("metrics", MetricsOut.ok(metrics, overlays, cue), json=True)

    except Exception as e:
        # print full traceback in the server console AND send message to client
        traceback.print_exc()
        emit("metrics", MetricsOut.paused(f"{type(e).__name__}: {e}"), json=True)

@socketio.on("frame_ping")
def frame_ping(_):
    emit("pong", {"ok": True})

@socketio.on("finish_session")
def finish_session(_):
    report = session_store.finish()
    emit("session", report, json=True)

if __name__ == "__main__":
    socketio.run(app, host=cfg["server"]["host"], port=cfg["server"]["port"])
