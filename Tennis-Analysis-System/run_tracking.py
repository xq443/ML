import os
import cv2
import csv
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from deep_sort_realtime.deepsort_tracker import DeepSort

def run_tracking(frames_dir, output_csv, fps):
    model = YOLO("yolo12n.pt")
    tracker = DeepSort(max_age=30)
    frame_files = sorted(Path(frames_dir).glob("*.jpg"))
    results = []

    tracked_id = None  # Will hold the player ID to track

    for frame_id, frame_path in enumerate(frame_files):
        timestamp = frame_id / fps
        frame = cv2.imread(str(frame_path))

        detections = model(frame)[0]
        boxes = []
        confidences = []

        for box in detections.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            cls = int(box.cls)
            if cls == 0:  # person class
                boxes.append([x1, y1, x2, y2])
                confidences.append(conf)

        print(f"Frame {frame_id}:")
        print("  Boxes:", boxes)
        print("  Confidences:", confidences)

        dummy_feature = np.zeros(128, dtype=np.float32)
        detections_for_deepsort = [
            (box, conf, dummy_feature) for box, conf in zip(boxes, confidences)
        ]

        tracks = tracker.update_tracks(detections_for_deepsort, frame=frame)

        # Debug print all confirmed tracks this frame
        confirmed_ids = [t.track_id for t in tracks if t.is_confirmed()]
        print(f"  Confirmed Track IDs this frame: {confirmed_ids}")

        for track in tracks:
            if not track.is_confirmed():
                continue

            if tracked_id is None:
                tracked_id = track.track_id
                print(f"Tracking player with ID: {tracked_id}")

            if track.track_id != tracked_id:
                continue  # Only track the chosen player

            l, t, r, b = track.to_ltrb()
            x_pixel = int((l + r) / 2)
            y_pixel = int((t + b) / 2)

            results.append({
                "frame_id": frame_id,
                "timestamp": round(timestamp, 2),
                "player_id": track.track_id,
                "x_pixel": x_pixel,
                "y_pixel": y_pixel,
                "bbox_w": int(r - l),
                "bbox_h": int(b - t),
            })

    if results:
        # Ensure output directory exists
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

        keys = results[0].keys()
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"Tracking results saved to {output_csv}")
        print(f"Total tracked entries: {len(results)}")
    else:
        print("No tracking results to save.")

if __name__ == "__main__":
    run_tracking(
        frames_dir="extracted_frames",
        output_csv="extracted_frames/player_tracking.csv",
        fps=10.0
    )
