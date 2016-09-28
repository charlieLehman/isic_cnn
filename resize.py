import cv2
import numpy as np
import os.path as op
import json
from tqdm import tqdm as bar
from collections import Counter

import utils 
from rgb2binary import rgb2binary

dataset_path = utils.workingDirectory.get()
dataset = utils.dataset.load_json()


utils.imageSet.process_all(utils.imageSet.to_HSV,'_HSV')
utils.imageSet.process_all(utils.imageSet.to_DCT,'_DCT')
utils.imageSet.process_all(utils.imageSet.to_DCT_of_HSV,'_DCT_of_HSV')

benign_path = op.join(dataset_path,'images/benign')
malign_path = op.join(dataset_path,'images/malignant')

