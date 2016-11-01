from multiprocessing import Pool
from os import path
from tqdm import tqdm
import time
import cv2
import utils as u

def path_list():
    data = u.isic_dataset.load_json()
    data = [x for x in data if x['b_m'] != 'unknown']
    img_path = []
    for n in range(0,len(data)):
        img_path.append(path.join(data[n]['fileLoc'],data[n]['filename']))
    return img_path

def process_image(img_path):
    tag = '_DCT' 
    try:
        im = cv2.imread(img_path+'_256.png', cv2.IMREAD_ANYCOLOR)
        proc_image = u.imageSet.to_DCT(im)
        cv2.imwrite(img_path + tag +'.png',proc_image )
    except cv2.error as e:
        print(img_path)

if __name__ == '__main__':
    pool = Pool()
    l = path_list()
    rs = pool.imap(process_image, l)
    for n in tqdm(l):
        rs.next()
