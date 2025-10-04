# utils.py
def valid_keypoints(kps: dict, cfg) -> bool:
    if not kps: return False
    min_vis = cfg["confidence"]["min_visibility"]
    needed = ["left_shoulder","right_shoulder","left_hip","right_hip"]
    return all(k in kps and kps[k][3] >= min_vis for k in needed)
