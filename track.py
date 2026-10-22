from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
from algo0 import algo0


VIDEO   = "YTB_1791655562357 360P.mp4"
OUT_RAW = "output.mp4"

tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.3)

cap = cv2.VideoCapture(VIDEO)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS) or 30
out = cv2.VideoWriter(OUT_RAW, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

n = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    # detect
    mask = algo0(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        detections.append(([x, y, w, h], 1.0, "egg"))

    # track
    tracks = tracker.update_tracks(detections, frame=frame)

    # draw
    for t in tracks:
        if not t.is_confirmed() or t.time_since_update > 0:
            continue
        l, tp, r, b = map(int, t.to_ltrb())
        cv2.rectangle(frame, (l, tp), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f"egg {t.track_id}", (l, tp - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    out.write(frame)
    n += 1

cap.release()
out.release()
print("frames written:", n)