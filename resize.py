import cv2
import numpy as np

img = cv2.imread('/matlab/ISIC_0000054.jpg')

height, width = img.shape[:2]
res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)

imshow(res)
