import utils as u
import cv2
import numpy as np


im = cv2.imread('/tmp/isic_dataset/images/benign/ISIC_0000000_HSV_32.png')


pickled_im = u.cifar.pickle(im, 'benign')

unpickled_im = u.cifar.unpickle(pickled_im)
cv2.imshow('image',im)
cv2.imshow('pickled_image',unpickled_im)
cv2.waitKey(0)
cv2.destroyAllWindows()

