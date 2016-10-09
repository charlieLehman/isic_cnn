from multiprocessing import Pool
from os import path
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
    tag = '_DCT_block_multi' 
    if path.isfile(img_path+'_32.png'):
        try:
            im = cv2.imread(img_path+'_32.png', cv2.IMREAD_ANYCOLOR)
            proc_image = u.imageSet.to_HSV(im)
            cv2.imwrite(img_path + tag +'.png',proc_image )
            #cv2.imwrite(img_path + tag +'_32.png', imageSet.resize_to_32(proc_image))
        except cv2.error as e:
            print(data[n]['id'])
    else:
        try:
            im = cv2.imread(img_path+'.jpg', cv2.IMREAD_ANYCOLOR)
            cv2.imwrite(img_path +'_32.png', u.imageSet.resize_to_32(im))
            im_32 = cv2.imread(img_path+'_32.png', cv2.IMREAD_ANYCOLOR)
            proc_image = u.imageSet.to_HSV(im_32)
            cv2.imwrite(img_path + tag +'.png',proc_image )
        except cv2.error as e:
            print(data[n]['id'])

if __name__ == '__main__':
    pool = Pool(processes=4)              # process per core
    rs = pool.map_async(process_image, path_list())  # proces data_inputs iterable with pool
    while (True):
      if (rs.ready()): break
      remaining = rs._number_left
      print("Waiting for", remaining, "tasks to complete...")
      time.sleep(0.5)
