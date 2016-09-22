import numpy as np
import cv2

def rgb2binary(image,label):
    im = cv2.imread(image)
    im = (np.array(im))
    r = im[:,:,0].flatten()
    g = im[:,:,1].flatten()
    b = im[:,:,2].flatten()
    return np.array(list(int(label)) + list(r) + list(g) + list(b),np.uint8)
