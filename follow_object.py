import cv2
import argparse
import numpy as np

def apply_mask(image, mask_binary):
    # Set mask color
    color = np.array([0,255,0], dtype='uint8')
    # Apply mask to the image
    masked_img = np.where(mask_binary[...,None], color, image)
    out = cv2.addWeighted(image, 0.8, masked_img, 0.2,0)
    
    return out

# -----
# RGB image
image = cv2.imread('./DAVIS/JPEGImages/480p/bear/00009.jpg')
mask = cv2.imread('./DAVIS/Annotations/480p/bear/00009.png')
# Binary mask
_, mask_binary = cv2.threshold(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
print(image.shape)
print(mask_binary.shape)
# -----

out = apply_mask(image, mask_binary)
cv2.imshow('Masked object', out)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
