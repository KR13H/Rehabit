from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class FrameIn(BaseModel):
    img_b64: str
    mode: str

class MetricsOut(BaseModel):
    status: str
    metrics: Dict[str, Any] = {}
    overlays: Dict[str, Any] = {}
    cue: Optional[str] = None

    @staticmethod
    def ok(metrics, overlays, cue=None):
        return MetricsOut(status="ok", metrics=metrics, overlays=overlays, cue=cue).model_dump()
    @staticmethod
    def paused(msg):
        return MetricsOut(status="paused", metrics={"message": msg}).model_dump()
