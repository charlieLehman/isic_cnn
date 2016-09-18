from PIL import Image
import numpy as np
import random as ran
import sys
from termcolor import colored, cprint
import json
import os.path
from pprint import pprint
from tqdm import tqdm

import isic_api as api

api.checkstate()

img_dataset_path = '/tmp/isic_dataset/'
b_m_dict = {'benign':0,'malignant':1}

def imprep(dataset):
    im_dir = dataset.get('b_m')+'/'+dataset.get('filename')+'.jpg'
    im = Image.open(im_dir)
    im = (np.array(im))
    r = im[:,:,0].flatten()
    g = im[:,:,1].flatten()
    b = im[:,:,2].flatten()
    label = [b_m_dict[dataset.get('b_m')]]
    return np.array(list(label) + list(r) + list(g) + list(b),np.uint8)

    
if os.path.exists(img_dataset_path):
    with open(img_dataset_path + 'img_dataset.json') as img_dataset:
        data = json.load(img_dataset)
    out = []
    bin_meta = []
    meta_out = []
    size = 8
    while size:
        index = ran.randrange(len(data))
        im_choice = data[index]
        if im_choice not in bin_meta:
            bin_meta.append(dict(im_choice))
            out = np.append(imprep(im_choice),out)
            size = size-1
    mal_data = [x for x in data if x['b_m'] == 'malignant']
    size = 2
    while size:
        index = ran.randrange(len(mal_data))
        im_choice = mal_data[index]
        if im_choice not in bin_meta:
            bin_meta.append(dict(im_choice))
            out = np.append(imprep(im_choice),out)
            size = size-1
    out.tofile('out.bin')

    f=open('out.json','w')
    f.write(json.dumps(bin_meta))
    pprint(json.dumps(bin_meta))
    f.close


else:
    cprint('There is no dataset')
    sys.exit(1)




