import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
                           
# PATHS (change these)
SPLIT_CSV  = "split.csv"
IMAGES_DIR = "folder"
MASKS_DIR  = "sam_masks"
BAD_DIRS   = ["bad_mask/devs", "bad_mask/eval"] 
OUT_DIR    = "results"
os.makedirs(OUT_DIR, exist_ok=True)


# ALGO0: blur -> threshold ->
def algo0(image, thresh=170, min_area=5000):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
    return mask


#  METRICS: how much the prediction overlaps the SAM mask
def metrics(gt, pred):
    gt, pred = gt > 0, pred > 0
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    return {"iou":       tp / max(tp + fp + fn, 1),
            "dice":      2 * tp / max(2 * tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1),
            "recall":    tp / max(tp + fn, 1)}


# RUN ON EVERY IMAGE IN split.csv
bad = {os.path.splitext(f)[0] for d in BAD_DIRS for f in os.listdir(d)}
rows, shown = [], {}

for r in pd.read_csv(SPLIT_CSV).itertuples():
    name = os.path.splitext(r.filename)[0]
    if name in bad:
        continue
    img = cv2.imread(f"{IMAGES_DIR}/{r.filename}")
    gt  = cv2.imread(f"{MASKS_DIR}/{name}.png", 0)
    if img is None or gt is None:
        print("skipped:", r.filename)
        continue
    pred = algo0(img)
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    rows.append({"filename": r.filename, "split": r.split, **metrics(gt, pred)})
    shown[r.filename] = (img, gt, pred)

df = pd.DataFrame(rows)
df.to_csv(f"{OUT_DIR}/algo0_per_image.csv", index=False)


# OVERALL: average per split 
overall = df.groupby("split")[["iou", "dice", "precision", "recall"]].mean().round(4)
overall.insert(0, "images", df.groupby("split").size())
overall.to_csv(f"{OUT_DIR}/algo0_overall.csv")
print(overall)


#  5 BEST / 5 WORST 
def save_png(d, path):
    fig, ax = plt.subplots(len(d), 3, figsize=(10, 3.2 * len(d)))
    for i, r in enumerate(d.itertuples()):
        img, gt, pred = shown[r.filename]
        ax[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[i, 0].set_title(f"{r.split} | IoU {r.iou:.2f}")
        ax[i, 1].imshow(gt, cmap="gray");   ax[i, 1].set_title("SAM mask")
        ax[i, 2].imshow(pred, cmap="gray"); ax[i, 2].set_title("algo0")
        for a in ax[i]: a.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()

ranked = df.sort_values("iou", ascending=False)
save_png(ranked.head(5),            f"{OUT_DIR}/best.png")
save_png(ranked.tail(5).iloc[::-1], f"{OUT_DIR}/worst.png")
print("saved:", sorted(os.listdir(OUT_DIR)))