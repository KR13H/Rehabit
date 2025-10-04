# app.py
import requests
import base64

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64, numpy as np, cv2, yaml, requests   # add requests, base64
from pose_backend.mediapipe_engine import PoseEngine
from pose_backend.schemas import FrameIn, MetricsOut
from pose_backend.utils import valid_keypoints
from trackers.arms import ArmsTracker
from trackers.sit_to_stand import SitStandTracker
from trackers.march import MarchTracker
from services.session_store import SessionStore

# Load config
cfg = yaml.safe_load(open("config/settings.yaml"))

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

pose_engine = PoseEngine(cfg)
trackers = {
    "arms": ArmsTracker(cfg),
    "sit":  SitStandTracker(cfg),
    "march": MarchTracker(cfg)
}
session_store = SessionStore()

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("frame")
def on_frame(data):
    mode = data.get("mode", "arms")
    img_b64 = data["img_b64"].split(",")[1]
    img = base64.b64decode(img_b64)
    frame = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)

    kps = pose_engine.infer(frame)
    if not valid_keypoints(kps, cfg):
        emit("metrics", MetricsOut.paused("Recenter your body"), json=True)
        return

    metrics, overlays, cue = trackers[mode].update(kps)

    if cue:
        # Gemini API call
        gem_cfg = cfg["services"]["gemini"]
        gem_payload = {
            "model": gem_cfg["model"],
            "prompt": f"The patient has this issue: {cue} The asymmetry index is {metrics.get('asymmetry_index', 0):.2f}. Provide one simple corrective tip."
        }
        gem_headers = {"Authorization": f"Bearer {gem_cfg['key']}"}
        gem_resp = requests.post(gem_cfg["url"], json=gem_payload, headers=gem_headers)
        tip_text = gem_resp.json().get("choices", [{}])[0].get("text", cue)

        # ElevenLabs TTS call
        eleven_cfg = cfg["services"]["elevenlabs"]
        tts_payload = {"text": tip_text}
        tts_headers = {
            "xi-api-key": eleven_cfg["key"],
            "Content-Type": "application/json"
        }
        tts_resp = requests.post(eleven_cfg["url"], json=tts_payload, headers=tts_headers)
        if tts_resp.status_code == 200:
            audio_b64 = base64.b64encode(tts_resp.content).decode()
        else:
            audio_b64 = ""

        # Send updates
        emit("metrics", MetricsOut.ok(metrics, overlays, cue=tip_text), json=True)
        if audio_b64:
            socketio.emit("audio_cue", {"audio": audio_b64})

    else:
        emit("metrics", MetricsOut.ok(metrics, overlays), json=True)


@socketio.on("frame")
def on_frame(data):
    mode = data.get("mode", "arms")
    img_b64 = data["img_b64"].split(",")[1]
    img = base64.b64decode(img_b64)
    frame = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)

    kps = pose_engine.infer(frame)
    if not valid_keypoints(kps, cfg):
        emit("metrics", MetricsOut.paused("Recenter your body"), json=True)
        return

    metrics, overlays, cue = trackers[mode].update(kps)

    # If trackers returned a cue (e.g. asymmetry warning), enrich it
    if cue:
        # 1) Gemini text generation
        gem_cfg = cfg["services"]["gemini"]
        gem_payload = {
            "model": gem_cfg["model"],
            "prompt": f"{cue} Patient asymmetry index: {metrics['asymmetry_index']:.2f}. Provide a single corrective tip in simple words."
        }
        gem_headers = {"Authorization": f"Bearer {gem_cfg['key']}"}
        gem_resp = requests.post(gem_cfg["url"], json=gem_payload, headers=gem_headers)
        tip_text = gem_resp.json().get("choices", [{}])[0].get("text", cue)

        # 2) ElevenLabs TTS
        eleven_cfg = cfg["services"]["elevenlabs"]
        tts_payload = {"text": tip_text}
        tts_headers = {
            "xi-api-key": eleven_cfg["key"],
            "Content-Type": "application/json"
        }
        tts_resp = requests.post(eleven_cfg["url"], json=tts_payload, headers=tts_headers)
        audio_b64 = base64.b64encode(tts_resp.content).decode()

        # Emit enriched feedback
        emit("metrics", MetricsOut.ok(metrics, overlays, cue=tip_text), json=True)
        socketio.emit("audio_cue", {"audio": audio_b64})
    else:
        # No special cue, just send metrics
        emit("metrics", MetricsOut.ok(metrics, overlays), json=True)

@socketio.on("finish_session")
def finish_session(data):
    report = session_store.finish()
    emit("session", report, json=True)

if __name__ == "__main__":
    socketio.run(app, host=cfg["services"].get("host","0.0.0.0"), port=cfg["services"].get("port",5000))
