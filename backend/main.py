from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os, uuid, json
import numpy as np
import cv2
from nova_client import nova_coach_feedback
import mediapipe as mp
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

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
def compute_quality(frames_kps: list, min_v: float = 0.5):
    """
    Simple quality: fraction of frames where core joints are visible.
    """
    core = ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip"]
    good = 0
    for f in frames_kps:
        if not f:
            continue
        ok = True
        for k in core:
            if kp_v(f, k) < min_v:
                ok = False
                break
        if ok:
            good += 1
    return good / max(1, len(frames_kps))

def estimate_camera_angle(frames_kps: list):
    """
    Very rough: if shoulder width (pixel distance between L/R shoulders) is large -> more side-ish.
    """
    widths = []
    for f in frames_kps:
        ls = kp_xy(f, "l_shoulder")
        rs = kp_xy(f, "r_shoulder")
        if ls and rs:
            widths.append(abs(ls[0] - rs[0]))
    if not widths:
        return "unknown"
    w_med = float(np.median(widths))
    # heuristics: tune later
    if w_med > 160:
        return "side_view"
    if w_med > 90:
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
            if forward < 0.06:
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
            if disp < 0.06:
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
    quality = compute_quality(frames_kps)
    cam = estimate_camera_angle(frames_kps)

    contact_i, speeds = find_contact_frame(frames_kps, handedness)

    # If quality is low, return early with guidance (prevents bad advice)
    if quality < 0.35 or cam == "unknown":
        return {
            "ok": False,
            "saved": fname,
            "quality_score": quality,
            "camera_angle": cam,
            "message": "Pose tracking quality is low. Please re-record: full body in frame, good lighting, side view.",
        }

    if cam == "front_or_back":
        # allow but warn (or you can hard-fail)
        warning = "Camera angle looks front/back. Side view (90°) is recommended for best accuracy."
    else:
        warning = None

    if shot_type == "forehand":
        metrics, flags = analyze_forehand(frames_kps, contact_i, handedness, meta)
        perfect = {
            "focus": "Forehand",
            "targets": [
                "Contact point in front of the body.",
                "Elbow angle ~70–100° near contact (approx).",
                "Wrist not dropping too low relative to hips.",
                "Clear follow-through forward and upward.",
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
            ],
        }
    else:
        return {"ok": False, "saved": fname, "message": "Invalid shot_type"}

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
        "camera_angle": cam,
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