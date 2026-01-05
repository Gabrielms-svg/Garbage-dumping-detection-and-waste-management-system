from ultralytics import YOLO
import cv2
import time
import csv
import os
from datetime import datetime

# ------------------ HELPERS ------------------

def overlaps(box1, box2):
    x1,y1,x2,y2 = box1
    a1,b1,a2,b2 = box2
    return not (x2 < a1 or x1 > a2 or y2 < b1 or y1 > b2)

# ------------------ MODELS ------------------

waste_model = YOLO("best.pt")
vehicle_model = YOLO("yolov8n.pt")


cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS)
W, H = int(cap.get(3)), int(cap.get(4))

# ------------------ CONFIG ------------------

CSV_FILE = "dumping_events.csv"
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # login/
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")

GROUND_RATIO = 0.55
WASTE_PERSIST_TIME = 2.0
ACTOR_LEAVE_TIME = 1.0
RESET_DELAY = 8.0
VIDEO_DURATION = 10  # seconds

os.makedirs(EVIDENCE_DIR, exist_ok=True)

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerow([
            "event_id", "timestamp", "video",
            "actor", "evidence_video"
        ])

# ------------------ STATE ------------------

event_id = 0
dump_active = False
actor_seen_once = False

actor_last_seen = 0
waste_first_seen = 0
waste_last_seen = 0

recording = False
video_writer = None
record_start_time = 0

# ------------------ LOOP ------------------

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
        event_id += 1
        recording = True
        record_start_time = now

        event_dir = os.path.join(EVIDENCE_DIR, f"event_{event_id}")
        os.makedirs(event_dir, exist_ok=True)

        video_path = os.path.join(event_dir, "dumping.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H))

        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow([
                event_id, ts, "10.mp4",
                actor_label, video_path
            ])

        print(f"[RECORDING] Dumping event {event_id}")

    # ------------------ WRITE VIDEO ------------------

    if recording:
        video_writer.write(frame)

        if (now - record_start_time) >= VIDEO_DURATION:
            recording = False
            video_writer.release()
            video_writer = None
            print(f"[SAVED] 10s evidence video for event {event_id}")

    # ------------------ HARD RESET ------------------

    if dump_active:
        if not waste_boxes and (now - waste_last_seen) > RESET_DELAY:
            dump_active = False
            actor_seen_once = False
            waste_first_seen = 0
            print("[RESET] Scene cleared")

    # ------------------ DISPLAY ------------------

    cv2.imshow("Illegal Dumping Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
