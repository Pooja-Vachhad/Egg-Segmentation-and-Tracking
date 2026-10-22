import cv2
import numpy as np   

# ALGO0: threshold + contours 
def algo0(image, thresh=170, min_area=5000, blur_k=5, sigma=0, save_path=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_k, blur_k), sigma)
    _, th = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)

    if save_path:                        
        cv2.imwrite(save_path, mask)
    return mask


img = cv2.imread("egg.jpg")
mask = algo0(img, save_path="egg_mask.png")   
        