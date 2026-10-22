from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np


# Input video and output file names
VIDEO   = "YTB_1791655562357 360P.mp4"
OUT_RAW = "output.mp4"


# Same as the dataset algo0, but min_area = 500 (eggs look smaller in the video)
def algo0(image, thresh=170, min_area=500, blur_k=5, sigma=0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_k, blur_k), sigma)
    _, th = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
    return mask


# Create the DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.3)

# Open the video and prepare the output video writer
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

    # 1. Detect: run algo0, then make one box for each white blob
    mask = algo0(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        detections.append(([x, y, w, h], 1.0, "egg"))

    # 2. Track: DeepSORT gives each box an ID and keeps it across frames
    tracks = tracker.update_tracks(detections, frame=frame)

    # 3. Draw: only confirmed tracks that were seen in this frame
    for t in tracks:
        if not t.is_confirmed() or t.time_since_update > 0:
            continue
        l, tp, r, b = map(int, t.to_ltrb())
        cv2.rectangle(frame, (l, tp), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f"egg {t.track_id}", (l, tp - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    out.write(frame)
    n += 1

    if n % 100 == 0:
        print("frame", n)

cap.release()
out.release()
print("frames written:", n)
