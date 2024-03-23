import cv2
import argparse
import numpy as np

def applyThresholdingAndShow(frame, bbox):
    if bbox[2] == 0 or bbox[3] == 0:
        print("Nebyla vybrána žádná oblast. Ukončení programu.")
        return None  # Vrátí None, pokud nebyla vybrána žádná oblast
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    roi = frame[y:y+h, x:x+w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresholded_roi = cv2.threshold(roi_gray, 127, 255, cv2.THRESH_BINARY)
    black_background = np.zeros_like(frame)
    black_background[y:y+h, x:x+w, 0] = thresholded_roi
    black_background[y:y+h, x:x+w, 1] = thresholded_roi
    black_background[y:y+h, x:x+w, 2] = thresholded_roi
    return black_background

image = cv2.imread('./DAVIS/JPEGImages/480p/bear/00000.jpg')
if image is None:
    print("Obrázek nebyl nalezen. Zkontrolujte cestu k souboru.")
else:
    bbox = cv2.selectROI("Tracking", image, False)
    cv2.destroyAllWindows()
    
    if bbox[2] > 0 and bbox[3] > 0:
        thresholded_image_on_black = applyThresholdingAndShow(image, bbox)
        if thresholded_image_on_black is not None:
            cv2.imshow('Thresholded Image on Black Background', thresholded_image_on_black)
            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()
    else:
        print("Nebyla vybrána žádná oblast.")
cv2.destroyAllWindows()
