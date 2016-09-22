import cv2
import numpy as np
import os.path as op
import json
from tqdm import tqdm as bar
from collections import Counter

import utils as api
from rgb2binary import rgb2binary

dataset_path = api.state.get()
dataset = json.load(open(op.join(dataset_path, 'dataset.json')))
count_benign = 0 
count_malignant = 0

def imageTo32(image):
    try:
        res = cv2.resize(im,(32,32), interpolation = cv2.INTER_CUBIC)
        return res
    except cv2.error as e:
        print(e)

for n in bar(range(0,len(dataset))):
    for x in ['benign', 'malignant']:
        if dataset[n]['b_m']==x:
            count_benign += 1
            filename = op.join(dataset_path,'images',x, dataset[n]['filename'])
            if op.isfile(filename + '.jpg'):
                try:
                    im = cv2.imread(filename + '.jpg')
                    cv2.imwrite(filename + '_32.jpg', imageTo32(im))
                except cv2.error as e:
                    print(dataset[n]['id'])
            else:
                print(dataset[n]['id'])


benign_path = op.join(dataset_path,'images/benign')
malign_path = op.join(dataset_path,'images/malignant')

