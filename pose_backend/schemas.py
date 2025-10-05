# pose_backend/schemas.py
from pydantic import BaseModel
from typing import Any, Dict, Optional

def _to_dict(model: BaseModel) -> Dict[str, Any]:
    # Works on both Pydantic v2 (model_dump) and v1 (dict)
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()

class MetricsOut(BaseModel):
    status: str
    metrics: Dict[str, Any] = {}
    overlays: Dict[str, Any] = {}
    cue: Optional[str] = None

    @staticmethod
    def ok(metrics, overlays, cue=None):
        return _to_dict(MetricsOut(status="ok", metrics=metrics, overlays=overlays, cue=cue))

    @staticmethod
    def paused(msg):
        return _to_dict(MetricsOut(status="paused", metrics={"message": msg}))
