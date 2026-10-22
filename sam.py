import cv2, os
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


folder_path = "folder"       
sam_folder  = "sam_folder"   


os.makedirs(sam_folder, exist_ok=True)
files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith((".jpeg", ".jpg", ".png"))])
print("images:", len(files))

# SAM 
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(device)
gen = SamAutomaticMaskGenerator(sam, points_per_side=15, pred_iou_thresh=0.88,
                                stability_score_thresh=0.92, min_mask_region_area=500)

# filter: keep only egg masks
def keep(m, H, W, gray, margin=3, shadow_factor=1.0):
    frac = m["area"] / (H * W)
    x, y, w, h = m["bbox"]
    touches_border = (x <= margin or y <= margin or
                      x + w >= W - margin or y + h >= H - margin)
    if not (0.03 < frac < 0.6) or touches_border:   # size rule + border rule
        return False
    return gray[m["segmentation"]].mean() > shadow_factor * gray.mean() 

# main loop 
for i, f in enumerate(files):
    img = cv2.cvtColor(cv2.imread(os.path.join(folder_path, f)), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    masks = gen.generate(img)

    final = np.zeros((H, W), np.uint8)
    kept = 0
    for m in masks:
        if keep(m, H, W, gray):
            final[m["segmentation"]] = 255   
            kept += 1

    mask_path = os.path.join(sam_folder, os.path.splitext(f)[0] + ".png")
    cv2.imwrite(mask_path, final)
