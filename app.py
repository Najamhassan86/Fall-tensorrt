import os
import time
import json
import re
import cv2
from flask import Flask, Response, render_template, request, send_from_directory, jsonify
from trt_yolo import YOLODetectorTRT
from tracker_iou import IoUTracker
from events import EventLogic
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()

# ---------------- CONFIG ----------------
ENGINE_PATH = os.environ.get("ENGINE_PATH", "model.engine")
CLASS_NAMES = os.environ.get("CLASS_NAMES", "falling,sitting,standing,walking,bending").split(",")

CAMERA_ID = int(os.environ.get("CAMERA_ID", "0"))
CAP_W = int(os.environ.get("CAP_W", "1280"))
CAP_H = int(os.environ.get("CAP_H", "720"))
CAP_FPS = int(os.environ.get("CAP_FPS", "30"))

CONF = float(os.environ.get("CONF", "0.5"))
IOU_NMS = float(os.environ.get("IOU_NMS", "0.45"))

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Email (optional)
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "465"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")

ALERT_TO_DEFAULT = os.environ.get("ALERT_TO", "")
SETTINGS_FILE = os.environ.get("SETTINGS_FILE", "./settings.json")

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ---------------- SETTINGS (persist alert_to) ----------------
def _load_settings():
    # auto-create settings.json if missing
    if not os.path.exists(SETTINGS_FILE):
        data = {"alert_to": ALERT_TO_DEFAULT}
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            return {"alert_to": ALERT_TO_DEFAULT}

    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"alert_to": ALERT_TO_DEFAULT}
        if "alert_to" not in data:
            data["alert_to"] = ALERT_TO_DEFAULT
        return data
    except Exception:
        return {"alert_to": ALERT_TO_DEFAULT}


def _save_settings(data: dict) -> bool:
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False


def get_alert_to() -> str:
    return str(_load_settings().get("alert_to", ALERT_TO_DEFAULT) or "")


def set_alert_to(email: str) -> (bool, str):
    email = (email or "").strip()
    if not email:
        return False, "Email is required."
    if not EMAIL_RE.match(email):
        return False, "Invalid email format."

    data = _load_settings()
    data["alert_to"] = email
    ok = _save_settings(data)
    if not ok:
        return False, "Failed to save settings.json (permissions?)."
    return True, "Saved."


def email_config_ok() -> bool:
    return bool(SMTP_HOST and SMTP_USER and SMTP_PASS and get_alert_to())


def send_email(subject, body):
    # NOTE: receiver is dynamic now (settings.json)
    alert_to = get_alert_to()
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and alert_to):
        return False

    msg = EmailMessage()
    msg["From"] = SMTP_USER
    msg["To"] = alert_to
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as s:
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        return True
    except Exception:
        return False


# ---------------- INIT ----------------
app = Flask(__name__)

detector = YOLODetectorTRT(ENGINE_PATH, CLASS_NAMES, conf_threshold=CONF, iou_threshold=IOU_NMS)
tracker = IoUTracker(iou_match=0.35, max_age_sec=1.0)

event_logic = EventLogic(
    fps=CAP_FPS,
    movement_threshold=20,
    fall_window=10,
    fall_required=6,
    enable_postfall=True,
    postfall_window_sec=5,
    lying_aspect_thresh=0.55,
    lying_frames_required=10,
    immobile_sec_required=3,
    sit_limit_min=10,
    alert_cooldown_sec=20,
)

last_status = {"fall_confirmed": False, "sit_too_long": False, "should_alert": False, "ts": 0.0}


def open_camera():
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_ID)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


cap = open_camera()

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    out_url = request.args.get("out", None)
    return render_template(
        "index.html",
        out_url=out_url,
        alert_to=get_alert_to(),
        email_ok=email_config_ok(),
        smtp_user=SMTP_USER,
        camera=f"/dev/video{CAMERA_ID}" if str(CAMERA_ID).isdigit() else str(CAMERA_ID),
        cap_w=CAP_W,
        cap_h=CAP_H,
        cap_fps=CAP_FPS,
        conf=CONF,
        iou_nms=IOU_NMS,
    )


@app.route("/status")
def status():
    return jsonify(last_status)


@app.route("/settings", methods=["GET"])
def settings_get():
    return jsonify({
        "alert_to": get_alert_to(),
        "email_ok": email_config_ok(),
        "smtp_user": SMTP_USER,
        "smtp_host": SMTP_HOST,
        "smtp_port": SMTP_PORT,
    })


@app.route("/settings", methods=["POST"])
def settings_set():
    # supports JSON or form POST
    email = ""
    if request.is_json:
        data = request.get_json(silent=True) or {}
        email = data.get("alert_to", "")
    else:
        email = request.form.get("alert_to", "")

    ok, msg = set_alert_to(email)
    return jsonify({
        "ok": ok,
        "message": msg,
        "alert_to": get_alert_to(),
        "email_ok": email_config_ok(),
    }), (200 if ok else 400)


@app.route("/send_test_email", methods=["POST"])
def send_test_email():
    if not email_config_ok():
        return jsonify({"ok": False, "message": "Email not configured correctly (check .env + receiver)."}), 400

    ok = send_email("✅ Test Alert (Fall System)", "This is a test email from your Jetson fall detection system.")
    return jsonify({"ok": bool(ok), "message": "Sent." if ok else "Failed to send (check SMTP/app password)."}), (200 if ok else 500)


@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("video")
    if not f:
        return "No file", 400

    in_path = os.path.join(UPLOAD_DIR, f"{int(time.time())}_{f.filename}")
    f.save(in_path)

    out_name = f"out_{int(time.time())}.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    process_video(in_path, out_path)
    return render_template(
        "index.html",
        out_url=f"/outputs/{out_name}",
        alert_to=get_alert_to(),
        email_ok=email_config_ok(),
        smtp_user=SMTP_USER,
        camera=f"/dev/video{CAMERA_ID}" if str(CAMERA_ID).isdigit() else str(CAMERA_ID),
        cap_w=CAP_W,
        cap_h=CAP_H,
        cap_fps=CAP_FPS,
        conf=CONF,
        iou_nms=IOU_NMS,
    )


def process_video(in_path, out_path):
    vcap = cv2.VideoCapture(in_path)
    fps = int(vcap.get(cv2.CAP_PROP_FPS)) or 30
    w = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # separate logic for offline run
    logic = EventLogic(
        fps=fps,
        movement_threshold=event_logic.movement_threshold,
        fall_window=event_logic.fall_buf.maxlen,
        fall_required=event_logic.fall_required,
        enable_postfall=event_logic.enable_postfall,
        postfall_window_sec=event_logic.postfall_window_sec,
        lying_aspect_thresh=event_logic.lying_aspect_thresh,
        lying_frames_required=event_logic.lying_frames_required,
        immobile_sec_required=event_logic.immobile_sec_required,
        sit_limit_min=(event_logic.sit_limit_sec / 60.0),
        alert_cooldown_sec=event_logic.alert_cooldown_sec,
    )
    local_tracker = IoUTracker(iou_match=0.35, max_age_sec=1.0)

    while True:
        ret, frame = vcap.read()
        if not ret:
            break

        dets = detector.detect(frame)
        tracked = local_tracker.update(dets)
        ev = logic.update(tracked)

        frame = detector.draw(frame, tracked)
        if ev["fall_confirmed"]:
            cv2.putText(frame, "FALL CONFIRMED!", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        if ev["sit_too_long"]:
            cv2.putText(frame, "SITTING TOO LONG!", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        out.write(frame)

    vcap.release()
    out.release()


@app.route("/video_feed")
def video_feed():
    def gen():
        global cap, last_status

        while True:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                time.sleep(0.2)
                cap = open_camera()
                continue

            dets = detector.detect(frame)
            tracked = tracker.update(dets)
            ev = event_logic.update(tracked)

            last_status = {**ev, "ts": time.time()}

            # email alert only on "should_alert"
            if ev["should_alert"]:
                send_email("⚠️ FALL ALERT", "A confirmed fall has been detected.")

            frame = detector.draw(frame, tracked)

            cv2.putText(frame, f"Detections: {len(tracked)}", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
            if ev["fall_confirmed"]:
                cv2.putText(frame, "FALL CONFIRMED!", (12, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)

