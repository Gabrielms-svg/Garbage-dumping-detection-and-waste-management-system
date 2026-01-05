import os
import json
from datetime import datetime
from camera_registry import CAMERA_REGISTRY

BASE_DIR = "evidence"

def create_event(event_id, rtsp_url):
    cam = CAMERA_REGISTRY.get(rtsp_url)

    if not cam:
        raise ValueError(f"Camera not registered: {rtsp_url}")

    event_dir = os.path.join(BASE_DIR, f"event_{event_id}")
    os.makedirs(f"{event_dir}/dumping", exist_ok=True)
    os.makedirs(f"{event_dir}/plates", exist_ok=True)

    data = {
        "event_id": event_id,
        "camera": {
            "id": cam["camera_id"],
            "rtsp": rtsp_url,
            "location": cam["location"],
            "ward": cam["ward"],
            "city": cam["city"]
        },
        "dumping": {},
        "plates": []
    }

    with open(f"{event_dir}/event.json", "w") as f:
        json.dump(data, f, indent=4)

    return event_dir


def update_dumping_video(event_dir, video_path):
    _update(event_dir, lambda d: d.update({
        "dumping": {
            "video": video_path,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }))


def add_plate(event_dir, plate_path):
    _update(event_dir, lambda d: d["plates"].append({
        "image": plate_path,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }))


def _update(event_dir, updater):
    path = f"{event_dir}/event.json"
    with open(path, "r+") as f:
        data = json.load(f)
        updater(data)
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()
