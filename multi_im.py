from multiprocessing import Pool
from os import path
from tqdm import tqdm
import time
import cv2
import utils as u

def path_list():
    data = u.isic_dataset.load_json()
    img_path = []
    for n in range(0,len(data)):
        img_path.append(path.join(data[n]['fileLoc'],data[n]['filename']))
    return img_path

def process_image(img_path):
    tag = '_DCT_block' 
    try:
        im = cv2.imread(img_path+'.jpg', cv2.IMREAD_ANYCOLOR)
        proc_image = u.imageSet.to_DCT_block(im)
        cv2.imwrite(img_path + tag +'.png',proc_image )
    except cv2.error as e:
        print(img_path)

if __name__ == '__main__':
    pool = Pool(processes=1)
    rs = pool.imap(process_image, path_list())
    for n in tqdm(path_list()):
        rs.next()