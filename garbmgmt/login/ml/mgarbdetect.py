from ultralytics import YOLO
import cv2
import time
import csv
import os
import json
from datetime import datetime

# ------------------ CAMERA CONFIG (ADMIN DEFINED) ------------------

CAMERA_ID = "cam_01"
CAMERA_LOCATION = "MG Road, Kochi"
RTSP_URL = "rtsp://localhost:8554/cam1"   # for future use

# ------------------ PATH SETUP (IMPORTANT FIX) ------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # login/
EVIDENCE_ROOT = os.path.join(BASE_DIR, "evidence")
os.makedirs(EVIDENCE_ROOT, exist_ok=True)

# ------------------ HELPERS ------------------

def overlaps(box1, box2):
    x1,y1,x2,y2 = box1
    a1,b1,a2,b2 = box2
    return not (x2 < a1 or x1 > a2 or y2 < b1 or y1 > b2)

# ------------------ MODELS ------------------

waste_model = YOLO("best.pt")
vehicle_model = YOLO("yolov8n.pt")

# ------------------ VIDEO SOURCE ------------------

cap = cv2.VideoCapture(RTSP_URL)  # later RTSP

fps = cap.get(cv2.CAP_PROP_FPS) or 25
W, H = int(cap.get(3)), int(cap.get(4))

# ------------------ CONFIG ------------------

GROUND_RATIO = 0.55
WASTE_PERSIST_TIME = 2.0
ACTOR_LEAVE_TIME = 1.0
RESET_DELAY = 8.0
VIDEO_DURATION = 10  # seconds

# ------------------ STATE ------------------

dump_active = False
actor_seen_once = False

actor_last_seen = 0
waste_first_seen = 0
waste_last_seen = 0

recording = False
video_writer = None
record_start_time = 0

current_event_dir = None

# ------------------ MAIN LOOP ------------------

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    vehicle_boxes = []
    waste_boxes = []
    actor_label = ""

    # ------------------ VEHICLE DETECTION ------------------

    veh_results = vehicle_model(frame, conf=0.4, verbose=False)

    for box in veh_results[0].boxes:
        label = vehicle_model.names[int(box.cls[0])]
        if label in ["car", "truck", "bus", "motorcycle"]:
            actor_seen_once = True
            actor_last_seen = now
            actor_label = label

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            vehicle_boxes.append((x1,y1,x2,y2))

    # ------------------ WASTE DETECTION ------------------

    waste_results = waste_model(frame, conf=0.25, verbose=False)

    for box in waste_results[0].boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0])

        if y2 < H * GROUND_RATIO:
            continue

        if any(overlaps((x1,y1,x2,y2), vb) for vb in vehicle_boxes):
            continue

        waste_boxes.append((x1,y1,x2,y2))
        waste_last_seen = now

        if waste_first_seen == 0:
            waste_first_seen = now

    # ------------------ CONFIRM DUMPING ------------------

    dumping_confirmed = (
        not dump_active and
        actor_seen_once and
        waste_boxes and
        (now - actor_last_seen) > ACTOR_LEAVE_TIME and
        (now - waste_first_seen) >= WASTE_PERSIST_TIME
    )

    # ------------------ START RECORDING ------------------

    if dumping_confirmed:
        dump_active = True
        recording = True
        record_start_time = now

        event_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        current_event_dir = os.path.join(
            EVIDENCE_ROOT,
            CAMERA_ID,
            f"event_{event_id}"
        )

        dump_dir = os.path.join(current_event_dir, "dumping")
        os.makedirs(dump_dir, exist_ok=True)

        video_path = os.path.join(dump_dir, "dumping.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H))

        # -------- SAVE METADATA --------
        metadata = {
            "event_id": event_id,
            "camera_id": CAMERA_ID,
            "location": CAMERA_LOCATION,
            "timestamp": ts,
            "actor": actor_label,
            "dumping_video": "dumping/dumping.mp4",
            "plates": []
        }

        with open(os.path.join(current_event_dir, "event.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"[RECORDING] Dumping event {event_id}")

    # ------------------ WRITE VIDEO ------------------

    if recording:
        video_writer.write(frame)

        if (now - record_start_time) >= VIDEO_DURATION:
            recording = False
            video_writer.release()
            video_writer = None
            print("[SAVED] 10s dumping evidence")

    # ------------------ RESET ------------------

    if dump_active:
        if not waste_boxes and (now - waste_last_seen) > RESET_DELAY:
            dump_active = False
            actor_seen_once = False
            waste_first_seen = 0
            print("[RESET] Scene cleared")

    # ------------------ DISPLAY ------------------

    cv2.imshow("Illegal Dumping Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
