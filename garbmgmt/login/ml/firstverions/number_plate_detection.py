from ultralytics import YOLO
import cv2
import os
import time
from datetime import datetime
import math

# ---------------- CONFIG ----------------
MODEL_PATH = "number_plate_detection.pt"
VIDEO_PATH = "10.mp4"
SAVE_DIR = "plate_evidence"

CONF_THRES = 0.4
IOU_THRES = 0.3
DIST_THRES = 80          # pixels
SAVE_COOLDOWN = 2.0      # seconds
SCALE_FACTOR = 2.5

# ---------------- SETUP ----------------
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

os.makedirs(SAVE_DIR, exist_ok=True)

saved_tracks = []   # [x1,y1,x2,y2,time]
event_id = 0

# ---------------- HELPERS ----------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)

def center_dist(a, b):
    ax = (a[0]+a[2])//2
    ay = (a[1]+a[3])//2
    bx = (b[0]+b[2])//2
    by = (b[1]+b[3])//2
    return math.hypot(ax-bx, ay-by)

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF_THRES, verbose=False)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        current_box = [x1,y1,x2,y2]
        now = time.time()

        matched = False

        for prev in saved_tracks:
            prev_box, prev_time = prev[:4], prev[4]

            if (iou(current_box, prev_box) > IOU_THRES or
                center_dist(current_box, prev_box) < DIST_THRES):

                # same vehicle, update box + time
                prev[:4] = current_box
                prev[4] = now
                matched = True
                break

        if matched:
            continue   # 🚫 DO NOT SAVE AGAIN

        # ----------- NEW VEHICLE -----------
        event_id += 1
        saved_tracks.append([*current_box, now])

        # Expand box
        h, w, _ = frame.shape
        pad_x = int((x2-x1)*0.35)
        pad_y = int((y2-y1)*0.45)

        x1e = max(0, x1-pad_x)
        y1e = max(0, y1-pad_y)
        x2e = min(w, x2+pad_x)
        y2e = min(h, y2+pad_y)

        crop = frame[y1e:y2e, x1e:x2e]
        if crop.size == 0:
            continue

        enlarged = cv2.resize(
            crop, None,
            fx=SCALE_FACTOR, fy=SCALE_FACTOR,
            interpolation=cv2.INTER_CUBIC
        )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{SAVE_DIR}/vehicle_{event_id}_{ts}.jpg"
        cv2.imwrite(path, enlarged)

        print(f"[SAVED] Vehicle {event_id}")

        cv2.rectangle(frame, (x1e,y1e), (x2e,y2e), (0,255,0), 2)

    cv2.imshow("Number Plate Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
