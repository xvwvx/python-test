
import cv2
import numpy as np

image_flags = cv2.IMREAD_COLOR
# image_flags = cv2.IMREAD_GRAYSCALE

target_img = cv2.imread('img/tmp/IMG_0396.png', flags=image_flags)
template = cv2.imread('img/♥️.png', flags=image_flags)
h, w = template.shape[:2]

res = cv2.matchTemplate(target_img, template, cv2.TM_CCOEFF_NORMED)
threshold = .98
loc = np.where(res >= threshold)

if image_flags == cv2.IMREAD_GRAYSCALE:
    target_img = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR)

for pt in zip(*loc[::-1]):  # Switch collumns and rows
    print(pt)
    cv2.rectangle(target_img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)

cv2.imwrite('result.png', target_img)