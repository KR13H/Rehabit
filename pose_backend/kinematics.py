import numpy as np, math

def angle(a, b, c):
    ax, ay = a; bx, by = b; cx, cy = c
    v1, v2 = np.array([ax-bx, ay-by]), np.array([cx-bx, cy-by])
    denom = (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6)
    cosang = np.clip(np.dot(v1, v2)/denom, -1, 1)
    return math.degrees(math.acos(cosang))

def trunk_angle_deg(mid_shoulder, mid_hip):
    # angle of body segment relative to vertical
    sx, sy = mid_shoulder[:2]; hx, hy = mid_hip[:2]
    return angle((sx, sy-100), (sx, sy), (hx, hy))

def asymmetry_index(left_val, right_val):
    return abs(left_val - right_val) / max(left_val, right_val, 1e-6)
