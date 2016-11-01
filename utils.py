import cv2
import json
import sys
import os.path as op
import os
import shutil
import requests
import numpy as np
import collections as c
from multiprocessing import Process, Pool
from oct2py import Oct2Py
from tqdm import tqdm
from termcolor import colored, cprint

class workingDirectory:
#User input to use default or custom path to store dataset
    def init():
        if workingDirectory.exists():
            current_dataset = json.load(open('.workingDirectory'))
            cprint("Your data is in "+ current_dataset['dir']+"\n","blue")
            dataset_path = current_dataset['dir']
        elif ui.ynQuery("Set up default path?\n /tmp/isic_dataset/"):
            dataset_path = '/tmp/isic_dataset/'
        else:
            dir_query = colored("Where should I set up the root directory for the dataset?\n", "blue")
            while True:
                try:
                    dataset_path = input(dir_query)
                    op.isdir(dataset_path)
                    dataset_path = op.join(dataset_path, 'isic_dataset')
                    break
                except FileNotFoundError:
                    colored("Please input a valid directory.\n","yellow")

        if op.exists(dataset_path):
            if ui.ynQuery("It looks like you already have the dataset. "
                                "Should I overwrite? "):
                return dataset_path
            else:
                sys.exit(1)
        else:
            return dataset_path

    def save(directory):
        f = open('.workingDirectory', 'w')
        workingDirectory = {}
        workingDirectory['dir'] = directory
        json.dump(workingDirectory,f)
        f.close()

    def exists():
        return op.isfile('.workingDirectory')

    def delete():
        if op.isfile('.workingDirectory'):
            current_dataset = json.load(open('.workingDirectory'))
            if ui.ynQuery("Should I delete " + current_dataset['dir']):
                shutil.rmtree(current_dataset['dir'])
                os.remove('.workingDirectory')

    def get():
        if workingDirectory.exists():
            current_dataset = json.load(open('.workingDirectory'))
            return current_dataset['dir']
        else:
            cprint("You need to run setup.py", "red")
class isic_api:
#ISIC-Archive API url
    isic_url = 'https://isic-archive.com:443/api/v1/image'

#Generates a JSON file for the dataset
    def get_id_list(directory):
#Get user input for the number of entries to collect
        while True:
            try:
                payload_limit = int(input(colored('How many entries should I ask for?\n', 
                                                  'blue')))
                break
            except ValueError:
                cprint('Please input an integer', 'red')

        isic_payload = {'limit': payload_limit, 'offset': '0'}

#Try to grab the list of IDs from ISIC
        try:
            isic_request = requests.get(isic_api.isic_url, params=isic_payload)
        except requests.exceptions.RequestException as e:
            cprint('There seems to be an issue with connecting...\n', 'red')
            return "Error: {}".format(e)
            sys.exit(1)

        img_id_list = isic_request.json()
        img_meta = {}
        dataset = []
        count_images = len(img_id_list)

        cprint('Accessing isic-archive\'s API to retreive dataset ID list. This will take a while...', 'blue')

#Iteratively query the API for each ID
        for n in tqdm(range(0, count_images)):
            img_meta_request = requests.get(isic_api.isic_url + '/' + img_id_list[n]['_id'])
            assert isinstance(img_meta_request, object)
            img_meta_list = img_meta_request.json()
            img_meta['id'] = img_id_list[n]['_id']
            img_meta['filename'] = img_meta_list['name']
            img_meta['fileLoc'] = op.join(directory, 'images', img_meta_list['meta']['clinical'].get('benign_malignant'))
            img_meta['age'] = img_meta_list['meta']['clinical'].get('age')
            img_meta['sex'] = img_meta_list['meta']['clinical'].get('sex')
            img_meta['b_m'] = img_meta_list['meta']['clinical'].get('benign_malignant')
            img_meta['diagnosis'] = img_meta_list['meta']['clinical'].get('diagnosis')
            if img_meta not in dataset and img_meta['b_m'] != 'unknown':
                dataset.append(dict(img_meta))

#Save the dataset to a directory
        if not op.exists(directory):
            os.makedirs(directory)
        f = open(op.join(directory, 'dataset.json'), 'w')
        f.write(json.dumps(dataset))
        f.close()
        cprint('Dataset is in ' + op.abspath(directory), 'blue')
        workingDirectory.save(directory)

#Get the images from ISIC archive
    def get_images(directory):
        while True:
            try:
                image_size = int(input(colored('What size of square image in pixels?\n', 'blue')))
                break
            except ValueError: 
                cprint('Please input an integer', 'red')
        try:
            with open(op.join(directory,'dataset.json')) as dataset:
                data = json.load(dataset)
            isic_api.__save_images__(directory, image_size, 'benign', [x for x in data if x['b_m'] == 'benign'])
            isic_api.__save_images__(directory, image_size, 'malignant', [x for x in data if x['b_m'] == 'malignant'])
        except FileNotFoundError as e:
            cprint('There is nothing here..\n', 'red')
            print(e)
            sys.exit(1)

#Saves the image for get_images
    def __save_images__(directory, image_size, name, id_set):
        isic_payload = {'width':image_size}
        print('Getting '+ name +' images \n')
        imdir = op.join(directory,'images',name)
        if not op.exists(imdir):
            os.makedirs(imdir)
        for n in tqdm(range(0,len(id_set))):
            img_name = id_set[n]['filename']
            with open(op.join(imdir,img_name)+'.jpg','wb') as f:
                if image_size > 512:
                    isic_request = requests.get(isic_api.isic_url+'/'+id_set[n]['id']+'/download')
                    f.write(isic_request.content)
                else:
                    isic_request = requests.get(isic_api.isic_url+'/'+id_set[n]['id']+'/thumbnail', params=isic_payload)
                    f.write(isic_request.content)

class ui:
    def ynQuery(question, default="no"):
        valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
        if default is None:
            prompt = " [y/n] "
        elif default == "yes":
            prompt = " [Y/n] "
        elif default == "no":
            prompt = " [y/N] "
        else:
            raise ValueError("invalid default answer: '%s'" % default)
        while True:
            cprint(question + prompt, 'yellow')
            choice = input().lower()
            if default is not None and choice == '':
                return valid[default]
            elif choice in valid:
                return valid[choice]
            else:
                cprint("Please respond with 'yes' or 'no' "
                       "(or 'y' or 'n').\n", "red")

class isic_dataset:                
    def load_json():
        data =  json.load(open(op.join(workingDirectory.get(),'dataset.json')))
        data = [x for x in data if x['b_m'] != 'unknown']
        return data

    def key_list():
        data =  isic_dataset.load_json() 
        return set( keys for dic in data for keys in dic.keys())

    def dict_to_list(dict_for_work,key):
        result_list = []
        for n in range(0,len(dict_for_work)):
            result_list.append(dict_for_work[n][key])
        return result_list

    def shuffle(list1, list2, row_len1, row_len2):
        result_list = []
        max_num_bins = int(min(np.floor(len(list1)/row_len1),np.floor(len(list2)/row_len2)))
        for n in range(1,max_num_bins+1):
            seg1 = list1[0+row_len1*(n-1):row_len1*n]
            seg2 = list2[0+row_len2*(n-1):row_len2*n]
            result_list.append(seg1+seg2)
        return result_list

class imageSet:
    def process_all_32(function, tag):
        """Operate on all downloaded images and save with appended tag
        """
        data = isic_dataset.load_json()
        for n in tqdm(range(0,len(data))):
            img_path = op.join(data[n]['fileLoc'],data[n]['filename'])
            if op.isfile(img_path+'_32.png'):
                try:
                    im = cv2.imread(img_path+'_32.png', cv2.IMREAD_ANYCOLOR)
                    proc_image = function(im)
                    cv2.imwrite(img_path + tag +'.png',proc_image )
                except cv2.error as e:
                    print(data[n]['id'])
            else:
                try:
                    im = cv2.imread(img_path+'.jpg', cv2.IMREAD_ANYCOLOR)
                    cv2.imwrite(img_path +'_32.png', imageSet.resize_to(im,32))
                    im_32 = cv2.imread(img_path+'_32.png', cv2.IMREAD_ANYCOLOR)
                    proc_image = function(im_32)
                    cv2.imwrite(img_path + tag +'.png',proc_image )
                except cv2.error as e:
                    print(data[n]['id'])

    def process_all(function, tag, *arg):
        """Operate on all downloaded images and save with appended tag
        """
        data = isic_dataset.load_json()
        for n in tqdm(range(0,len(data))):
            img_path = op.join(data[n]['fileLoc'],data[n]['filename'])
            if op.isfile(img_path+'.jpg'):
                try:
                    im = cv2.imread(img_path+'.jpg', cv2.IMREAD_ANYCOLOR)
                    proc_image = function(im, *arg)
                    cv2.imwrite(img_path + tag +'.png',proc_image )
                except cv2.error as e:
                    print(data[n]['id'])
            else:
                print(data[n]['id'])

    def resize_to(image, size):
        """Convert any size image to a 32x32 image 
        """
        try:
            return cv2.resize(image,(size,size), interpolation = cv2.INTER_CUBIC)
        except cv2.error as e:
            print(e)

    def to_HSV(image):
        """Convert image to a HSV 
        """
        try:
            return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        except cv2.error as e:
            print(e)

    def to_FFT(image):
        """Convert to FFT of RGB channels
        """
        fftim = np.zeros(np.shape(image))
        try:
            for n in [0,1,2]:
                freqim = np.fft.fft2(image[:,:,n])
                shiftim = np.fft.fftshift(freqim)
                fftim[:,:,n] = 20*np.log(np.abs(shiftim))
            return fftim

        except cv2.error as e:
            print(e)

    def to_FFT_of_HSV(image):
        """Convert to HSV of RGB channels
        """
        image = imageSet.to_HSV(image)
        fftim = np.zeros(np.shape(image))
        try:
            for n in [0,1,2]:
                freqim = np.fft.fft2(image[:,:,n])
                shiftim = np.fft.fftshift(freqim)
                fftim[:,:,n] = 20*np.log(np.abs(shiftim))
            return fftim

        except cv2.error as e:
            print(e)

    def to_DCT_block(image):
        """Convert to DCT of RGB channels
        """
        oct_inst = Oct2Py()
        try:
            oct_inst.eval('pkg load all')
            oct_inst.addpath('m_code')
            dct8im = oct_inst.block_dct(image,16)
            return dct8im
        except Exception as e:
            return str(e)

    def to_DCT(image):
        """Convert to DCT of RGB channels
        """
        h, w, d = np.shape(image)
        dctim = np.zeros([h,w,d])
        block = 8
        x = int(h/block)
        y = int(w/block)

        try:
            for n in [0,1,2]:
                for k in range(1,x):
                    for l in range(1,y):
                        dctim[block*k-block:block*k,block*l-block:block*l,n] = np.uint8(cv2.dct(np.float32(image[block*k-block:block*k,block*l-block:block*l,n])/255.0))*255.0

            return dctim

        except cv2.error as e:
            print(e)

    def to_DCT_flip(image):
        """Convert to DCT of RGB channels
        """
        h, w, d = np.shape(image)
        dctim = np.zeros([h,w,d])
        x = int(h/4)
        y = int(w/4)

        quad = [(0, 127, 0, 127,1), (128, 255, 0, 127,2), (0, 127, 128, 255,3) ,(128, 255, 128, 255,4)]
        def q1(im):
            return np.flipud(np.fliplr(im))
        def q2(im):
            return np.fliplr(im)
        def q3(im):
            return np.flipud(im)
        def q4(im):
            return im

            
        flipper = {1:q1,
                   2:q2,
                   3:q3,
                   4:q4}
        try:
            for n in [0,1,2]:
                for x_b, x_e, y_b, y_e, q in quad:
                    dctim[x_b:x_e,y_b:y_e,n] = flipper[q](np.uint8(cv2.dct(np.float32(image[x_b:x_e,y_b:y_e,n])/255.0))*255.0)

            return dctim

        except cv2.error as e:
            print(e)

    def to_DCT_of_HSV(image):
        """Convert to DCT of RGB channels
        """
        image = imageSet.to_HSV(image)
        try:
            dctim = imageSet.to_DCT(image)
            return dctim

        except cv2.error as e:
            print(e)

class probability:
    def distribution(key):
        data=isic_dataset.load_json() 
        sample_size = len(data)
        vals=[]
        dist=[]
        for n in range(0,sample_size):
            vals.append(data[n][key])
        dist = dict(c.Counter(vals))
        dist.update((k,v/sample_size) for k,v in dist.items())
        return dist

class cifar:
    def bin_list_gen():
        data = isic_dataset.load_json()
        benign_dict = [x for x in data if x['b_m'] == 'benign']
        malign_dict = [x for x in data if x['b_m'] == 'malignant']

        ben_id = isic_dataset.dict_to_list(benign_dict, 'id')
        mal_id = isic_dataset.dict_to_list(malign_dict, 'id')
        id_RV = isic_dataset.shuffle(ben_id,mal_id,465,35)
        with open(op.join(workingDirectory.get(),'id_RV.json'),'w') as f:
            json.dump(id_RV,f)
            f.close()

    def pickle(im, b_m):
        """Label + serialized image
        """
        label = {'benign':0, 'malignant':1}
        im = np.array(im)
        r = im[:,:,0].flatten()
        g = im[:,:,1].flatten()
        b = im[:,:,2].flatten()

        return np.array([label[b_m]] + list(r) + list(g) + list(b),np.uint8)
    
    def unpickle(pickle):
        """Assumes square image
           Mostly used for testing
        """
        channel_length = int((len(pickle)-1)/3)
        image_size =  int(np.sqrt(channel_length))
        flat_image = np.delete(pickle,0)
        im = np.zeros([image_size, image_size, 3])
        im[:,:,0] = flat_image[:channel_length].reshape([image_size, image_size])
        im[:,:,1] = flat_image[channel_length:channel_length*2].reshape([image_size, image_size])
        im[:,:,2] = flat_image[channel_length*2:].reshape([image_size, image_size])
        return np.array(im, np.uint8)

    def make_binary(tag):
        """Generates a CIFAR-like binary from a list of IDs
        """
        cifar.bin_list_gen()
        with open(op.join(workingDirectory.get(),'id_RV.json'),'r') as f:
            id_RV = json.load(f)
            f.close()

        data = isic_dataset.load_json()
        for index, n in enumerate(tqdm(id_RV)):
            binary = []
            for m in tqdm(n):
                x = [x for x in data if x['id'] == m]
                img_path = op.join(x[0]['fileLoc'],x[0]['filename'])
                im = cv2.imread(img_path+tag+'.png', cv2.IMREAD_ANYCOLOR)
                binary = np.array(list(binary)+list(cifar.pickle(im,x[0]['b_m'])))
            with open(op.join(workingDirectory.get(),'bin'+tag+'_'+str(index)+'.bin'),'wb') as f:
                f.write(binary)
                f.close()
        
class test:
    def data_vs_img():
        mismatch = []
        data = isic_dataset.load_json()
        for n in tqdm(range(0,len(data))):
            img_path = op.join(data[n]['fileLoc'],data[n]['filename'])
            if op.isfile(img_path+'_32.png') != True:
                mismatch.append(img_path)
        f = open('data_vs_img', 'w')
        json.dump(mismatch,f)
        f.close()


