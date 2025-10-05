from trackers.arms import ArmsTracker

def test_arms_tracker():
    cfg = {"metrics": {"asymmetry_warn": 0.25}}
    tracker = ArmsTracker(cfg)
    # Simulate keypoints for symmetry and asymmetry
    kps = {
        "left_elbow": (30, 20),
        "left_shoulder": (35, 25),
        "left_hip": (35, 40),
        "right_elbow": (30, 20),
        "right_shoulder": (35, 25),
        "right_hip": (35, 40)
    }
    metrics, overlays, cue = tracker.update(kps)
    assert metrics["asymmetry_index"] == 0
    assert cue is None
