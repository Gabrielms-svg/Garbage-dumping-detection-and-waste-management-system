# evidence_manager.py
import os
import json
import uuid
from django.conf import settings
from django.utils.dateparse import parse_datetime
from django.core.files import File
from .models import DumpingEvent, Camera, LegalDumpingLocation

def sync_and_list_events(camera_id=None):
    base_dir = settings.EVIDENCE_ROOT
    cameras = [camera_id] if camera_id else os.listdir(base_dir)

    for cam_id in cameras:
        cam_path = os.path.join(base_dir, cam_id)
        if not os.path.isdir(cam_path):
            continue

        camera, _ = Camera.objects.get_or_create(camera_id=cam_id)

        for event_folder in os.listdir(cam_path):
            event_path = os.path.join(cam_path, event_folder)
            json_file = os.path.join(event_path, "event.json")
            if not os.path.exists(json_file):
                continue

            with open(json_file, "r") as f:
                data = json.load(f)

            event_id = data.get("event_id") or str(uuid.uuid4())
            if DumpingEvent.objects.filter(event_id=event_id).exists():
                continue

            timestamp = parse_datetime(data.get("timestamp"))
            actor_name = data.get("actor", "")

            dumping_event = DumpingEvent(
                event_id=event_id,
                camera=camera,
                timestamp=timestamp,
                actor=actor_name
            )

            # location
            location_name = data.get("location")
            if location_name:
                location_instance = LegalDumpingLocation.objects.filter(name=location_name).first()
                dumping_event.location = location_instance

            # video
            video_file_name = data.get("dumping_video") or "dumping.mp4"
            video_file_path = os.path.join(event_path, video_file_name)

            if os.path.exists(video_file_path):
                # Save video to MEDIA_ROOT
                media_subpath = f"dumping_videos/{cam_id}/{event_id}/{video_file_name}"
                full_media_path = os.path.join(settings.MEDIA_ROOT, media_subpath)
                os.makedirs(os.path.dirname(full_media_path), exist_ok=True)

                with open(video_file_path, "rb") as f:
                    dumping_event.dumping_video.save(media_subpath, File(f), save=False)

            dumping_event.save()

    # Return events
    if camera_id:
        camera = Camera.objects.filter(camera_id=camera_id).first()
        if not camera:
            return []
        return DumpingEvent.objects.filter(camera=camera).order_by("-timestamp")

    return DumpingEvent.objects.all().order_by("-timestamp")
