from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64, numpy as np, cv2, yaml
from pose_backend.mediapipe_engine import PoseEngine
from pose_backend.schemas import MetricsOut
from pose_backend.utils import valid_keypoints
from trackers.arms import ArmsTracker
from trackers.sit_to_stand import SitStandTracker
from trackers.march import MarchTracker
from services.session_store import SessionStore

cfg = yaml.safe_load(open("config/settings.yaml"))
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

pose_engine = PoseEngine(cfg)
trackers = {
    "arms":  ArmsTracker(cfg),
    "sit":   SitStandTracker(cfg),
    "march": MarchTracker(cfg),
}
session_store = SessionStore(persist_dir=str(ROOT / ".sessions"))

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
    # data: {"img_b64": "...", "mode": "arms|sit|march"}
    mode = data.get("mode", "arms")
    img_b64 = data["img_b64"].split(",")[1]
    img = base64.b64decode(img_b64)
    frame = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)

    kps = pose_engine.infer(frame)  # dict: {name: (x,y,z,vis)}
    if not valid_keypoints(kps, cfg):
        emit("metrics", MetricsOut.paused("Recenter your body"), json=True)
        return

    metrics, overlays, cue = trackers[mode].update(kps)
    socketio.emit("metrics", MetricsOut.ok(metrics, overlays, cue), json=True)

@socketio.on("frame_ping")
def frame_ping(_):
    emit("pong", {"ok": True})

@socketio.on("finish_session")
def finish_session(data):
    report = session_store.finish()
    emit("session", report, json=True)

if __name__ == "__main__":
    socketio.run(app, host=cfg["server"]["host"], port=cfg["server"]["port"])
