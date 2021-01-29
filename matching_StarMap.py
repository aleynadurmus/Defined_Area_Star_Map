import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('StarMap.png')
img2 = img.copy()
template = cv2.imread('Small_area_rotated.png')
template1 = cv2.imread('Small_area.png')
w, h = template.shape[:-1]
t, z = template1.shape[:-1]


methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF']

for meth in methods:
    img = img2.copy()
    img1 = img2.copy()
    method= eval(meth)
    res = cv2.matchTemplate(img,template,method)
    res1 = cv2.matchTemplate(img, template1, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
        top_left1 = min_loc1
    else:
        top_left = max_loc
        top_left1 = max_loc1
    bottom_right = (top_left[0] + w, top_left[1] + h)
    bottom_right1 = (top_left1[0] + t, top_left1[1] + z)
    cv2.rectangle(img,top_left, bottom_right, 255, 2)
    cv2.rectangle(img1, top_left1, bottom_right1, 255, 2)



    plt.subplot(224),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Small area rotated'), plt.xticks([]), plt.yticks([])
    plt.subplot(223),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Star Map'), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(res1, cmap='gray')
    plt.title('Matching Small area '), plt.xticks([]), plt.yticks([])
    plt.subplot(221), plt.imshow(img1, cmap='gray')
    plt.title('Detected Star Map'), plt.xticks([]), plt.yticks([])
    plt.plot(meth)
    plt.show()

