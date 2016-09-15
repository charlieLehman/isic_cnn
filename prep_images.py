from PIL import Image
import numpy as np
import random as ran
import sys
from termcolor import colored, cprint
import json
import os.path
from pprint import pprint
from tqdm import tqdm

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
    while len(bin_meta) < 10:
        im_choice = ran.choice(data)
        if im_choice not in bin_meta:
            bin_meta.append(dict(im_choice))
            out = np.append(imprep(im_choice),out)
    out.tofile('out.bin')

    f=open('out.json','w')
    f.write(json.dumps(bin_meta))
    pprint(json.dumps(bin_meta))
    f.close


else:
    cprint('There is no dataset')
    sys.exit(1)




