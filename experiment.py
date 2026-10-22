import cv2
import numpy as np



# Keep only large blobs (and optionally drop ones touching the border), return as a mask.
def keep_big(binary, min_area=5000, drop_border=False):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = binary.shape
    mask = np.zeros_like(binary)
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        if drop_border:
            x, y, cw, ch = cv2.boundingRect(c)
            if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1:
                continue
        cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
    return mask


# Segment objects using a chosen threshold method (otsu, adaptive, or fixed).
def segment_objects(image, method="otsu", thresh=170, blur_k=5, sigma=0,
                    block=51, C=-10, min_area=5000, drop_border=False):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_k, blur_k), sigma)

    if method == "otsu":
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block, C)
    elif method == "fixed":
        _, th = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError("method must be 'otsu', 'adaptive' or 'fixed'")

    return keep_big(th, min_area, drop_border)
