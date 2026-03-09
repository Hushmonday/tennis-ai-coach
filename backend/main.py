from dotenv import load_dotenv
import os
import warnings

# Reduce noisy native logs from MediaPipe / TFLite.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

# Suppress known protobuf deprecation warning from third-party dependencies.
warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase.GetPrototype\(\) is deprecated\..*",
    category=UserWarning,
)

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

import os, uuid, json
import numpy as np
import cv2
import mediapipe as mp

try:
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.set_stderrthreshold(absl_logging.ERROR)
except Exception:
    pass

from nova_client import nova_coach_feedback
load_dotenv()
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}


# (Optional) allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
FOLLOW_THROUGH_MIN_NORM = 0.015

# ---------- geometry helpers ----------
def angle_abc(a, b, c) -> float:
    """Angle ABC in degrees with B as vertex. a,b,c are (x,y)."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosv = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosv)))

def dist(a, b) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))

def avg_pt(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

# ---------- pose extraction ----------
LANDMARKS = {
    "l_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    "r_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    "l_elbow": mp_pose.PoseLandmark.LEFT_ELBOW.value,
    "r_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    "l_wrist": mp_pose.PoseLandmark.LEFT_WRIST.value,
    "r_wrist": mp_pose.PoseLandmark.RIGHT_WRIST.value,
    "l_hip": mp_pose.PoseLandmark.LEFT_HIP.value,
    "r_hip": mp_pose.PoseLandmark.RIGHT_HIP.value,
    "l_knee": mp_pose.PoseLandmark.LEFT_KNEE.value,
    "r_knee": mp_pose.PoseLandmark.RIGHT_KNEE.value,
    "l_ankle": mp_pose.PoseLandmark.LEFT_ANKLE.value,
    "r_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE.value,
}

def extract_pose_keypoints(video_path: str, target_fps: int = 12):
    """
    Returns:
      frames_kps: list[dict[name] -> dict(x,y,v)] in pixel coords
      frame_times: list[float] seconds
      meta: dict
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / target_fps)))

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_idx = 0

    frames_kps = []
    frame_times = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % step != 0:
                frame_idx += 1
                continue

            t = frame_idx / float(fps)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            kps = {}
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                for name, idx in LANDMARKS.items():
                    p = lm[idx]
                    kps[name] = {
                        "x": float(p.x * w),
                        "y": float(p.y * h),
                        "v": float(p.visibility),
                    }
            frames_kps.append(kps)
            frame_times.append(float(t))

            frame_idx += 1

    cap.release()
    meta = {"fps": float(fps), "width": w, "height": h, "sample_step": step, "target_fps": target_fps}
    return frames_kps, frame_times, meta

# ---------- handedness mapping ----------
def side(name: str) -> str:
    return "l_" if name.startswith("l_") else "r_"

def swap_lr_key(key: str) -> str:
    if key.startswith("l_"):
        return "r_" + key[2:]
    if key.startswith("r_"):
        return "l_" + key[2:]
    return key

def get_hit_side(handedness: str) -> str:
    # right-handed -> hitting arm is right side
    # left-handed  -> hitting arm is left side
    return "r" if handedness == "right" else "l"

def kp_xy(kps_frame: dict, key: str):
    p = kps_frame.get(key)
    if not p:
        return None
    return (p["x"], p["y"])

def kp_v(kps_frame: dict, key: str):
    p = kps_frame.get(key)
    return float(p["v"]) if p else 0.0

# ---------- quality & camera angle ----------
def compute_quality(frames_kps: list, min_v: float = 0.35):
    """
    More tolerant quality score:
    - frame_coverage: fraction of frames with >=6/8 core joints visible
    - joint_visibility_mean: average visible-joint ratio over frames
    - tracked_frame_ratio: fraction of frames with any landmarks detected
    """
    core = ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip"]
    total = max(1, len(frames_kps))

    tracked = 0
    covered = 0
    vis_sum = 0.0

    for f in frames_kps:
        if f:
            tracked += 1
        n_vis = sum(1 for k in core if kp_v(f, k) >= min_v)
        vis_ratio = n_vis / float(len(core))
        vis_sum += vis_ratio
        if n_vis >= 6:
            covered += 1

    frame_coverage = covered / total
    joint_visibility_mean = vis_sum / total
    tracked_frame_ratio = tracked / total
    quality_score = 0.6 * frame_coverage + 0.4 * joint_visibility_mean

    return {
        "quality_score": float(quality_score),
        "frame_coverage": float(frame_coverage),
        "joint_visibility_mean": float(joint_visibility_mean),
        "tracked_frame_ratio": float(tracked_frame_ratio),
        "min_visibility_threshold": float(min_v),
    }

def estimate_camera_angle(frames_kps: list):
    """
    Estimate camera angle from torso geometry.
    Uses shoulder-width / torso-height ratio so it is less sensitive to zoom/resolution:
      - smaller ratio => more side view (shoulders appear narrower)
      - larger ratio => more front/back view
    """
    ratios = []
    for f in frames_kps:
        ls = kp_xy(f, "l_shoulder")
        rs = kp_xy(f, "r_shoulder")
        lh = kp_xy(f, "l_hip")
        rh = kp_xy(f, "r_hip")
        if not (ls and rs and lh and rh):
            continue

        shoulder_w = abs(ls[0] - rs[0])
        sh_mid = avg_pt(ls, rs)
        hip_mid = avg_pt(lh, rh)
        torso_h = abs(sh_mid[1] - hip_mid[1])
        if torso_h < 1.0:
            continue

        ratios.append(shoulder_w / torso_h)

    if not ratios:
        return "unknown"

    r_med = float(np.median(ratios))
    # Heuristics; side-view should typically have a narrower shoulder profile.
    if r_med < 0.58:
        return "side_view"
    if r_med < 0.90:
        return "semi_side_view"
    return "front_or_back"

# ---------- contact frame by wrist speed ----------
def find_contact_frame(frames_kps: list, handedness: str):
    hit = get_hit_side(handedness)
    wrist_key = f"{hit}_wrist"
    speeds = []
    prev = None
    for i, f in enumerate(frames_kps):
        p = kp_xy(f, wrist_key)
        if p is None or kp_v(f, wrist_key) < 0.4:
            speeds.append(0.0)
            prev = None
            continue
        if prev is None:
            speeds.append(0.0)
        else:
            speeds.append(dist(prev, p))
        prev = p
    if len(speeds) == 0:
        return 0, speeds
    idx = int(np.argmax(speeds))
    return idx, speeds

# ---------- ball tracking ----------
def extract_ball_trajectory(video_path: str, target_fps: int = 24):
    """
    Simple classical CV ball tracking (no deep model):
    - foreground segmentation
    - contour filtering for small, near-round objects
    Returns trajectory summary + metrics in pixel/normalized units.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"ok": False, "reason": "cannot_open_video"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / float(target_fps))))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if w <= 0 or h <= 0:
        cap.release()
        return {"ok": False, "reason": "invalid_video_size"}

    back_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=24, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    frame_idx = 0
    sampled_n = 0
    detected_n = 0
    traj = []
    prev_pt = None
    max_area = max(24.0, 0.0009 * w * h)
    min_area = max(3.0, 0.00002 * w * h)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        sampled_n += 1
        t = frame_idx / float(fps)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        fg = back_sub.apply(blur)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=1)
        _, bw = cv2.threshold(fg, 210, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < min_area or area > max_area:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            if cw <= 1 or ch <= 1:
                continue
            aspect = cw / float(ch)
            if aspect < 0.5 or aspect > 1.8:
                continue
            peri = float(cv2.arcLength(c, True)) + 1e-6
            circ = float(4.0 * np.pi * area / (peri * peri))
            if circ < 0.2:
                continue
            cx = x + 0.5 * cw
            cy = y + 0.5 * ch
            candidates.append((cx, cy, area, circ))

        picked = None
        if candidates:
            if prev_pt is None:
                picked = max(candidates, key=lambda z: (z[3], -abs(z[2] - (min_area + max_area) * 0.5)))
            else:
                px, py = prev_pt
                picked = min(candidates, key=lambda z: (z[0] - px) ** 2 + (z[1] - py) ** 2 - 20.0 * z[3])

        if picked is not None:
            cx, cy, _, _ = picked
            detected_n += 1
            prev_pt = (cx, cy)
            traj.append({"t": float(t), "x": float(cx), "y": float(cy)})
        else:
            prev_pt = None

        frame_idx += 1

    cap.release()

    tracked_ratio = detected_n / max(1, sampled_n)
    if len(traj) < 6 or tracked_ratio < 0.12:
        return {
            "ok": False,
            "reason": "insufficient_track",
            "tracked_ratio": float(tracked_ratio),
            "points_count": len(traj),
            "sampled_frames": int(sampled_n),
        }

    speeds = []
    for i in range(1, len(traj)):
        dt = max(1e-6, traj[i]["t"] - traj[i - 1]["t"])
        dp = dist((traj[i]["x"], traj[i]["y"]), (traj[i - 1]["x"], traj[i - 1]["y"]))
        speeds.append(dp / dt)

    x_vals = np.array([p["x"] for p in traj], dtype=np.float32)
    y_vals = np.array([p["y"] for p in traj], dtype=np.float32)
    t_vals = np.array([p["t"] for p in traj], dtype=np.float32)
    t0 = float(t_vals[0])
    t_rel = t_vals - t0

    # Lateral curvature: residual from linear x(t)
    if len(traj) >= 3:
        coef = np.polyfit(t_rel, x_vals, deg=1)
        x_fit = np.polyval(coef, t_rel)
        curve = float(np.max(np.abs(x_vals - x_fit)) / max(1.0, float(w)))
    else:
        curve = 0.0

    # Approximate bounce: local maximum y (image y grows downward), then upward rebound.
    bounce_idx = int(np.argmax(y_vals))
    bounce_detected = False
    if 1 <= bounce_idx < len(y_vals) - 2:
        pre = float(y_vals[bounce_idx] - y_vals[max(0, bounce_idx - 1)])
        post = float(y_vals[min(len(y_vals) - 1, bounce_idx + 2)] - y_vals[bounce_idx])
        if pre > 1.5 and post < -1.0:
            bounce_detected = True

    return {
        "ok": True,
        "tracked_ratio": float(tracked_ratio),
        "points_count": len(traj),
        "sampled_frames": int(sampled_n),
        "first_t": float(traj[0]["t"]),
        "last_t": float(traj[-1]["t"]),
        "duration_s": float(traj[-1]["t"] - traj[0]["t"]),
        "avg_speed_px_s": float(np.mean(speeds)) if speeds else 0.0,
        "max_speed_px_s": float(np.max(speeds)) if speeds else 0.0,
        "lateral_curve_norm": curve,
        "vertical_drop_norm": float((y_vals[-1] - y_vals[0]) / max(1.0, float(h))),
        "bounce_detected": bool(bounce_detected),
        "bounce_t": float(traj[bounce_idx]["t"]),
        "bounce_x_norm": float(traj[bounce_idx]["x"] / max(1.0, float(w))),
    }

def add_ball_metrics_and_flags(
    shot_type: str,
    metrics: dict,
    flags: list,
    ball_track: dict,
    contact_t: float | None,
):
    if not ball_track.get("ok"):
        return metrics, flags

    metrics["ball_track_confidence"] = float(ball_track.get("tracked_ratio", 0.0))
    metrics["ball_lateral_curve_norm"] = float(ball_track.get("lateral_curve_norm", 0.0))
    metrics["ball_vertical_drop_norm"] = float(ball_track.get("vertical_drop_norm", 0.0))
    metrics["ball_avg_speed_px_s"] = float(ball_track.get("avg_speed_px_s", 0.0))

    if contact_t is not None and ball_track.get("bounce_detected"):
        bt = float(ball_track.get("bounce_t", 0.0))
        dt = bt - float(contact_t)
        if dt > 0:
            metrics["ball_contact_to_bounce_s"] = dt

    if ball_track.get("tracked_ratio", 0.0) < 0.25:
        flags.append("ball_track_low_confidence")

    conf = float(ball_track.get("tracked_ratio", 0.0))
    points_n = int(ball_track.get("points_count", 0))
    curve = float(ball_track.get("lateral_curve_norm", 0.0))
    # Only trust curve advice when track is sufficiently reliable.
    if conf >= 0.45 and points_n >= 12:
        if curve > 0.18:
            flags.append("ball_trajectory_very_curved")
        elif curve > 0.12:
            flags.append("ball_trajectory_curved")

    if shot_type == "serve":
        if not ball_track.get("bounce_detected"):
            flags.append("serve_bounce_not_visible")
        cb = metrics.get("ball_contact_to_bounce_s")
        if isinstance(cb, (int, float)) and cb > 0.95:
            flags.append("serve_depth_may_be_long")
        if isinstance(cb, (int, float)) and cb < 0.20:
            flags.append("serve_depth_may_be_short")

    return metrics, flags

def validate_shot_presence(
    shot_type: str,
    metrics: dict,
    quality_diag: dict,
    speeds: list,
    meta: dict,
):
    """
    Heuristic shot-presence validator.
    Returns:
      {present: bool, confidence: float, reasons: list[str]}
    """
    reasons = []
    evidence = 0.0

    width = max(1.0, float(meta.get("width", 1.0)))
    max_speed_norm = (max(speeds) / width) if speeds else 0.0
    q = float(quality_diag.get("quality_score", 0.0))

    if q >= 0.35:
        evidence += 0.20
    else:
        reasons.append("pose_quality_low_for_shot_detection")

    if max_speed_norm >= 0.02:
        evidence += 0.25
    else:
        reasons.append("racket_arm_motion_not_clear")

    if shot_type in ("forehand", "backhand"):
        if "elbow_angle_at_contact_deg" in metrics:
            evidence += 0.15
        if float(metrics.get("follow_through_disp_norm", 0.0)) >= 0.012:
            evidence += 0.20
        else:
            reasons.append("follow_through_signal_weak")

        if shot_type == "forehand":
            if abs(float(metrics.get("wrist_minus_hipcenter_normX", 0.0))) >= 0.02:
                evidence += 0.20
            else:
                reasons.append("forehand_contact_signal_weak")
        else:
            if float(metrics.get("wrist_to_hipcenter_norm", 0.0)) >= 0.08:
                evidence += 0.20
            else:
                reasons.append("backhand_contact_signal_weak")

    elif shot_type == "serve":
        if "wrist_vs_shoulder_normY_at_contact" in metrics:
            evidence += 0.15
            if float(metrics.get("wrist_vs_shoulder_normY_at_contact", 0.2)) <= 0.06:
                evidence += 0.10
        else:
            reasons.append("serve_contact_height_signal_missing")

        if "min_knee_angle_deg" in metrics:
            evidence += 0.15
        else:
            reasons.append("serve_knee_signal_missing")

        if float(metrics.get("hip_upward_drive_norm", 0.0)) >= 0.015:
            evidence += 0.15
        else:
            reasons.append("serve_upward_drive_signal_weak")

        if "ball_contact_to_bounce_s" in metrics:
            evidence += 0.15

    confidence = float(clamp(evidence, 0.0, 1.0))
    return {
        "present": confidence >= 0.55,
        "confidence": confidence,
        "reasons": reasons,
    }

# ---------- shot-specific metrics/flags ----------
def analyze_forehand(frames_kps: list, contact_i: int, handedness: str, meta: dict):
    hit = get_hit_side(handedness)
    sh, el, wr = f"{hit}_shoulder", f"{hit}_elbow", f"{hit}_wrist"
    lhip, rhip = "l_hip", "r_hip"

    f = frames_kps[contact_i] if contact_i < len(frames_kps) else {}
    s = kp_xy(f, sh); e = kp_xy(f, el); w = kp_xy(f, wr)
    lh = kp_xy(f, lhip); rh = kp_xy(f, rhip)

    metrics = {}
    flags = []

    if s and e and w:
        elbow_angle = angle_abc(s, e, w)
        metrics["elbow_angle_at_contact_deg"] = elbow_angle
        if elbow_angle < 55:
            flags.append("low_elbow")

    if w and s and lh and rh:
        hip_center = avg_pt(lh, rh)
        # y increases downward in image
        wrist_vs_shoulder = (w[1] - s[1]) / max(1.0, meta["height"])
        wrist_vs_hip = (w[1] - hip_center[1]) / max(1.0, meta["height"])
        metrics["wrist_vs_shoulder_normY"] = wrist_vs_shoulder
        metrics["wrist_vs_hip_normY"] = wrist_vs_hip
        # wrist much lower than hip center (positive means lower)
        if wrist_vs_hip > 0.08:
            flags.append("wrist_too_low")

    # contact in front of body (simple proxy using hip center x)
    if w and lh and rh:
        hip_center = avg_pt(lh, rh)
        metrics["wrist_minus_hipcenter_normX"] = (w[0] - hip_center[0]) / max(1.0, meta["width"])
        # For side view: for a right-hander filmed side-on, "in front" depends on facing direction.
        # We avoid guessing direction and only flag if extremely behind (close to torso).
        if abs(w[0] - hip_center[0]) < 0.03 * meta["width"]:
            flags.append("contact_too_close_to_body")

    # follow-through score: wrist movement after contact
    post_n = min(len(frames_kps) - contact_i - 1, int(meta["target_fps"] * 0.5))  # ~0.5s
    if post_n > 3 and w:
        w0 = w
        w_end = None
        for j in range(contact_i + 1, contact_i + 1 + post_n):
            fj = frames_kps[j]
            pj = kp_xy(fj, wr)
            if pj and kp_v(fj, wr) > 0.4:
                w_end = pj
        if w_end:
            forward = dist(w0, w_end) / max(1.0, meta["width"])
            metrics["follow_through_disp_norm"] = forward
            if forward < FOLLOW_THROUGH_MIN_NORM:
                flags.append("weak_follow_through")

    return metrics, flags

def analyze_backhand(frames_kps: list, contact_i: int, handedness: str, meta: dict):
    # treat backhand still as hitting side arm (dominant wrist)
    hit = get_hit_side(handedness)
    sh, el, wr = f"{hit}_shoulder", f"{hit}_elbow", f"{hit}_wrist"
    lhip, rhip = "l_hip", "r_hip"

    f = frames_kps[contact_i] if contact_i < len(frames_kps) else {}
    s = kp_xy(f, sh); e = kp_xy(f, el); w = kp_xy(f, wr)
    lh = kp_xy(f, lhip); rh = kp_xy(f, rhip)

    metrics = {}
    flags = []

    if w and lh and rh:
        hip_center = avg_pt(lh, rh)
        d_to_body = dist(w, hip_center) / max(1.0, meta["width"])
        metrics["wrist_to_hipcenter_norm"] = d_to_body
        if d_to_body < 0.10:
            flags.append("too_close_to_body")

    if s and e and w:
        elbow_angle = angle_abc(s, e, w)
        metrics["elbow_angle_at_contact_deg"] = elbow_angle
        if elbow_angle < 50:
            flags.append("arm_too_bent")

    # follow-through
    post_n = min(len(frames_kps) - contact_i - 1, int(meta["target_fps"] * 0.5))
    if post_n > 3 and w:
        w0 = w
        w_end = None
        for j in range(contact_i + 1, contact_i + 1 + post_n):
            pj = kp_xy(frames_kps[j], wr)
            if pj and kp_v(frames_kps[j], wr) > 0.4:
                w_end = pj
        if w_end:
            disp = dist(w0, w_end) / max(1.0, meta["width"])
            metrics["follow_through_disp_norm"] = disp
            if disp < FOLLOW_THROUGH_MIN_NORM:
                flags.append("weak_follow_through")

    return metrics, flags

def analyze_serve(frames_kps: list, times: list, contact_i: int, handedness: str, meta: dict):
    hit = get_hit_side(handedness)
    wr = f"{hit}_wrist"
    knee = f"{hit}_knee"
    hip = f"{hit}_hip"
    sh = f"{hit}_shoulder"
    el = f"{hit}_elbow"

    metrics = {}
    flags = []

    # knee bend: min knee angle around contact window
    # angle(hip, knee, ankle)
    ankle = f"{hit}_ankle"
    knee_angles = []
    for i, f in enumerate(frames_kps):
        h = kp_xy(f, hip); k = kp_xy(f, knee); a = kp_xy(f, ankle)
        if h and k and a and kp_v(f, knee) > 0.4:
            knee_angles.append(angle_abc(h, k, a))
        else:
            knee_angles.append(None)

    valid = [x for x in knee_angles if x is not None]
    if valid:
        min_knee = float(np.min(valid))
        metrics["min_knee_angle_deg"] = min_knee
        if min_knee > 155:  # barely bends
            flags.append("no_knee_bend")

    # contact height
    f = frames_kps[contact_i] if contact_i < len(frames_kps) else {}
    w = kp_xy(f, wr); s = kp_xy(f, sh)
    if w and s:
        # negative => wrist higher than shoulder
        rel = (w[1] - s[1]) / max(1.0, meta["height"])
        metrics["wrist_vs_shoulder_normY_at_contact"] = rel
        if rel > 0.02:
            flags.append("low_contact")

    # upward drive: hip y from min-knee frame to contact frame
    # find frame with min knee angle
    if valid:
        min_i = int(np.nanargmin([x if x is not None else np.nan for x in knee_angles]))
        fmin = frames_kps[min_i]
        fcon = frames_kps[contact_i]
        hip_min = kp_xy(fmin, hip)
        hip_con = kp_xy(fcon, hip)
        if hip_min and hip_con:
            up = (hip_min[1] - hip_con[1]) / max(1.0, meta["height"])  # positive means moved up
            metrics["hip_upward_drive_norm"] = up
            if up < 0.03:
                flags.append("no_upward_drive")

    # trophy-ish: elbow above shoulder near contact window (very rough)
    if f:
        e = kp_xy(f, el)
        if e and s:
            if e[1] > s[1]:  # elbow lower than shoulder (y bigger)
                flags.append("elbow_not_up")

    return metrics, flags

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def range_score(value: float, good_lo: float, good_hi: float, hard_lo: float, hard_hi: float) -> float:
    """
    0..1 score:
      - 1.0 in [good_lo, good_hi]
      - linearly decays to 0 at hard boundaries
    """
    if value <= hard_lo or value >= hard_hi:
        return 0.0
    if good_lo <= value <= good_hi:
        return 1.0
    if value < good_lo:
        return (value - hard_lo) / max(1e-8, good_lo - hard_lo)
    return (hard_hi - value) / max(1e-8, hard_hi - good_hi)

def low_is_better_score(value: float, good_max: float, hard_max: float) -> float:
    if value <= good_max:
        return 1.0
    if value >= hard_max:
        return 0.0
    return (hard_max - value) / max(1e-8, hard_max - good_max)

def high_is_better_score(value: float, good_min: float, hard_min: float) -> float:
    if value >= good_min:
        return 1.0
    if value <= hard_min:
        return 0.0
    return (value - hard_min) / max(1e-8, good_min - hard_min)

def metric_value(metrics: dict, key: str):
    v = metrics.get(key)
    if isinstance(v, (int, float)):
        return float(v)
    return None

def compute_technique_score(shot_type: str, metrics: dict):
    """
    Returns (0..1, details) from shot-specific objective metric bands.
    """
    details = {}
    weighted = 0.0
    total_w = 0.0

    def add(name: str, score: float, weight: float):
        nonlocal weighted, total_w
        s = clamp(score, 0.0, 1.0)
        weighted += s * weight
        total_w += weight
        details[name] = round(s, 3)

    if shot_type == "forehand":
        elbow = metric_value(metrics, "elbow_angle_at_contact_deg")
        if elbow is not None:
            add("elbow_angle", range_score(elbow, 75, 105, 45, 140), 0.28)

        wrist_hip = metric_value(metrics, "wrist_vs_hip_normY")
        if wrist_hip is not None:
            add("wrist_height_vs_hip", range_score(wrist_hip, -0.12, 0.03, -0.25, 0.12), 0.22)

        contact_x = metric_value(metrics, "wrist_minus_hipcenter_normX")
        if contact_x is not None:
            add("contact_distance_from_body", range_score(abs(contact_x), 0.06, 0.22, 0.01, 0.35), 0.22)

        follow = metric_value(metrics, "follow_through_disp_norm")
        if follow is not None:
            add("follow_through", high_is_better_score(follow, 0.04, 0.01), 0.28)

        curve = metric_value(metrics, "ball_lateral_curve_norm")
        if curve is not None:
            add("ball_flight_straightness", low_is_better_score(curve, 0.05, 0.20), 0.12)

    elif shot_type == "backhand":
        wrist_body = metric_value(metrics, "wrist_to_hipcenter_norm")
        if wrist_body is not None:
            add("contact_distance_from_body", range_score(wrist_body, 0.12, 0.30, 0.05, 0.38), 0.34)

        elbow = metric_value(metrics, "elbow_angle_at_contact_deg")
        if elbow is not None:
            add("elbow_extension", range_score(elbow, 75, 120, 45, 155), 0.30)

        follow = metric_value(metrics, "follow_through_disp_norm")
        if follow is not None:
            add("follow_through", high_is_better_score(follow, 0.04, 0.01), 0.36)

        curve = metric_value(metrics, "ball_lateral_curve_norm")
        if curve is not None:
            add("ball_flight_straightness", low_is_better_score(curve, 0.05, 0.20), 0.12)

    elif shot_type == "serve":
        knee = metric_value(metrics, "min_knee_angle_deg")
        if knee is not None:
            # smaller means more bend; around 125-155 is typically useful for recreational range
            add("knee_bend", range_score(knee, 125, 155, 105, 175), 0.34)

        contact_h = metric_value(metrics, "wrist_vs_shoulder_normY_at_contact")
        if contact_h is not None:
            add("contact_height", low_is_better_score(contact_h, -0.03, 0.08), 0.32)

        drive = metric_value(metrics, "hip_upward_drive_norm")
        if drive is not None:
            add("upward_drive", high_is_better_score(drive, 0.06, 0.015), 0.34)

        curve = metric_value(metrics, "ball_lateral_curve_norm")
        if curve is not None:
            add("serve_ball_straightness", low_is_better_score(curve, 0.04, 0.18), 0.22)

        ctob = metric_value(metrics, "ball_contact_to_bounce_s")
        if ctob is not None:
            add("serve_depth_timing", range_score(ctob, 0.30, 0.80, 0.12, 1.20), 0.18)

    if total_w <= 1e-8:
        # No reliable metric => neutral technique score
        return 0.5, {"note": "insufficient_metrics"}

    return weighted / total_w, details

def compute_overall_score(shot_type: str, quality_diag: dict, camera_angle: str, flags: list, metrics: dict):
    """
    Weighted objective score model (0-100):
      tracking 30 + camera 15 + technique 55 - penalties
    """
    tracking_score = (
        0.45 * clamp(quality_diag.get("quality_score", 0.0), 0.0, 1.0)
        + 0.35 * clamp(quality_diag.get("tracked_frame_ratio", 0.0), 0.0, 1.0)
        + 0.20 * clamp(quality_diag.get("frame_coverage", 0.0), 0.0, 1.0)
    ) * 30.0

    camera_score = {
        "side_view": 15.0,
        "semi_side_view": 10.0,
        "front_or_back": 4.0,
        "unknown": 0.0,
    }.get(camera_angle, 6.0)

    technique_ratio, technique_details = compute_technique_score(shot_type, metrics)
    technique_score = technique_ratio * 55.0

    flag_weights = {
        "contact_too_close_to_body": 4.0,
        "too_close_to_body": 4.0,
        "weak_follow_through": 5.0,
        "low_elbow": 4.0,
        "arm_too_bent": 4.0,
        "wrist_too_low": 3.0,
        "no_knee_bend": 6.0,
        "low_contact": 6.0,
        "no_upward_drive": 6.0,
        "elbow_not_up": 4.0,
        "ball_track_low_confidence": 2.0,
        "ball_trajectory_curved": 4.0,
        "ball_trajectory_very_curved": 7.0,
        "serve_bounce_not_visible": 2.0,
        "serve_depth_may_be_long": 4.0,
        "serve_depth_may_be_short": 4.0,
    }
    flag_penalty = 0.0
    for f in flags:
        flag_penalty += flag_weights.get(f, 3.0)
    flag_penalty = min(14.0, flag_penalty)

    # Mild regularization to avoid over-scoring poor tracking clips
    reliability_penalty = 0.0
    if quality_diag.get("tracked_frame_ratio", 0.0) < 0.35:
        reliability_penalty += 4.0
    if quality_diag.get("quality_score", 0.0) < 0.30:
        reliability_penalty += 2.0

    # Normalize components to 0..100 first, then combine with stable weights.
    tracking_norm = (tracking_score / 30.0) * 100.0
    camera_norm = (camera_score / 15.0) * 100.0
    technique_norm = technique_ratio * 100.0

    base_score = 0.30 * tracking_norm + 0.20 * camera_norm + 0.50 * technique_norm

    # Softer penalty model to avoid discouraging scores when overall movement is good.
    penalty = min(12.0, 0.40 * flag_penalty + 0.50 * reliability_penalty)
    bonus = 0.0
    if len(flags) <= 1 and quality_diag.get("quality_score", 0.0) >= 0.45:
        bonus += 4.0
    if len(flags) == 0 and quality_diag.get("quality_score", 0.0) >= 0.55:
        bonus += 3.0

    score = int(round(clamp(base_score - penalty + bonus, 0.0, 100.0)))

    return score, {
        "tracking_score": round(tracking_score, 2),
        "camera_score": round(camera_score, 2),
        "technique_score": round(technique_score, 2),
        "flag_penalty": round(flag_penalty, 2),
        "reliability_penalty": round(reliability_penalty, 2),
        "penalty_applied": round(penalty, 2),
        "bonus_applied": round(bonus, 2),
        "technique_details": technique_details,
    }

@app.get("/")
def root():
    return {"status": "backend running"}

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    handedness: str = Form(...),  # "right" | "left"
    shot_type: str = Form(...),   # "forehand" | "backhand" | "serve"
):
    ext = os.path.splitext(file.filename)[1] or ".webm"
    fname = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, fname)
    with open(path, "wb") as f:
        f.write(await file.read())

    frames_kps, times, meta = extract_pose_keypoints(path, target_fps=12)
    ball_track = extract_ball_trajectory(path, target_fps=24)
    quality_diag = compute_quality(frames_kps)
    quality = quality_diag["quality_score"]
    cam = estimate_camera_angle(frames_kps)

    contact_i, speeds = find_contact_frame(frames_kps, handedness)
    contact_t = float(times[contact_i]) if 0 <= contact_i < len(times) else None

    analysis_scope = {
        "pose_mechanics": True,
        "ball_trajectory": bool(ball_track.get("ok", False)),
        "serve_placement": bool(shot_type == "serve" and ball_track.get("ok", False) and ball_track.get("bounce_detected", False)),
    }

    # If quality is low, return early with guidance (prevents bad advice)
    if quality < 0.20 or quality_diag["tracked_frame_ratio"] < 0.15:
        coach_feedback = None
        nova_error = None
        try:
            coach_feedback = nova_coach_feedback(
                {
                    "shot_type": shot_type,
                    "handedness": handedness,
                    "camera_angle": cam,
                    "metrics": {},
                    "flags": ["low_pose_quality"],
                    "ball_track": ball_track,
                    "perfect_standard": {
                        "focus": shot_type,
                        "targets": ["Full body in frame", "Good lighting", "Side view (90°)"],
                    },
                }
            )
        except Exception as e:
            nova_error = str(e)

        return {
            "ok": False,
            "saved": fname,
            "quality_score": quality,
            # Score is not reliable when tracking quality is too low.
            "overall_score": None,
            "score_breakdown": {
                "tracking_score": 0,
                "camera_score": 0,
                "technique_score": 0,
                "flag_penalty": 0,
                "reliability_penalty": 0,
                "technique_details": {"note": "insufficient_quality"},
            },
            "quality_diagnostics": quality_diag,
            "camera_angle": cam,
            "analysis_scope": analysis_scope,
            "ball_track": ball_track,
            "message": "Pose tracking quality is low. Please re-record: full body in frame, good lighting, side view.",
            "coach_feedback": coach_feedback,
            "nova_error": nova_error,
        }

    warnings = []
    if cam == "front_or_back":
        warnings.append("Camera angle looks front/back. Side view (90°) is recommended for best accuracy.")
    if quality < 0.35:
        warnings.append("Pose tracking quality is moderate. Results may be less stable than usual.")
    if not ball_track.get("ok", False):
        warnings.append("Ball tracking confidence is low in this clip; ball-flight feedback may be limited.")
    elif shot_type == "serve" and not analysis_scope["serve_placement"]:
        warnings.append("Serve bounce/placement could not be confirmed from this clip.")
    warning = " ".join(warnings) if warnings else None

    if shot_type == "forehand":
        metrics, flags = analyze_forehand(frames_kps, contact_i, handedness, meta)
        perfect = {
            "focus": "Forehand",
            "targets": [
                "Contact point in front of the body.",
                "Elbow angle ~70–100° near contact (approx).",
                "Wrist not dropping too low relative to hips.",
                "Clear follow-through forward and upward.",
                "Ball exits with controlled, stable lateral path.",
            ],
        }
    elif shot_type == "backhand":
        metrics, flags = analyze_backhand(frames_kps, contact_i, handedness, meta)
        perfect = {
            "focus": "Backhand",
            "targets": [
                "Contact point in front of the body.",
                "Keep adequate extension (avoid being too close to torso).",
                "Smooth follow-through toward target direction.",
                "Ball flight stays stable with limited unwanted side curve.",
            ],
        }
    elif shot_type == "serve":
        metrics, flags = analyze_serve(frames_kps, times, contact_i, handedness, meta)
        perfect = {
            "focus": "Serve",
            "targets": [
                "Contact at a high point (near full reach).",
                "Noticeable knee bend and upward drive.",
                "Elbow up / trophy-like posture before acceleration.",
                "Ball path and bounce location are controlled and repeatable.",
            ],
        }
    else:
        return {"ok": False, "saved": fname, "message": "Invalid shot_type"}

    metrics, flags = add_ball_metrics_and_flags(shot_type, metrics, flags, ball_track, contact_t)
    # de-duplicate while preserving order
    flags = list(dict.fromkeys(flags))

    shot_detection = validate_shot_presence(shot_type, metrics, quality_diag, speeds, meta)
    if not shot_detection["present"]:
        return {
            "ok": False,
            "saved": fname,
            "quality_score": quality,
            "overall_score": None,
            "score_breakdown": {
                "tracking_score": 0,
                "camera_score": 0,
                "technique_score": 0,
                "flag_penalty": 0,
                "reliability_penalty": 0,
                "technique_details": {"note": "shot_not_detected"},
            },
            "quality_diagnostics": quality_diag,
            "camera_angle": cam,
            "analysis_scope": analysis_scope,
            "ball_track": ball_track,
            "shot_detection": shot_detection,
            "message": f"Selected shot type '{shot_type}' was not confidently detected in this video. Please upload a clearer {shot_type} clip.",
            "flags": ["shot_not_detected_or_mismatch"],
        }

    overall_score, score_breakdown = compute_overall_score(shot_type, quality_diag, cam, flags, metrics)

    # Step A output: keypoints sample (keep small)
    # Return only sparse frames to avoid huge payload
    sample_idx = list(range(0, len(frames_kps), max(1, len(frames_kps)//8)))[:8]
    keypoints_sample = [{"t": times[i], "kps": frames_kps[i]} for i in sample_idx]

        # ---------- Nova coach feedback ----------
    coach_feedback = None
    nova_error = None
    try:
        coach_feedback = nova_coach_feedback(
            {
                "shot_type": shot_type,
                "handedness": handedness,
                "camera_angle": cam,
                "metrics": metrics,
                "flags": flags,
                "analysis_scope": analysis_scope,
                "ball_track": ball_track,
                "perfect_standard": perfect,
            }
        )
    except Exception as e:
        nova_error = str(e)

    return {
        "ok": True,
        "saved": fname,
        "handedness": handedness,
        "shot_type": shot_type,
        "quality_score": quality,
        "overall_score": overall_score,
        "score_breakdown": score_breakdown,
        "quality_diagnostics": quality_diag,
        "camera_angle": cam,
        "analysis_scope": analysis_scope,
        "ball_track": ball_track,
        "shot_detection": shot_detection,
        "warning": warning,
        "meta": meta,
        "contact_frame_index": contact_i,
        "metrics": metrics,
        "flags": flags,
        "perfect_standard": perfect,
        "keypoints_sample": keypoints_sample,
        "coach_feedback": coach_feedback,
        "nova_error": nova_error,
    }
