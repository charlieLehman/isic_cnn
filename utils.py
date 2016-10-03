#Added a comment
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
from oct2py import octave
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
            if img_meta not in dataset:
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
        return json.load(open(op.join(workingDirectory.get(),'dataset.json')))

    def key_list():
        data =  isic_dataset.load_json() 
        return set( keys for dic in data for keys in dic.keys())

class imageSet:
    def process_all(function, tag):
        """Operate on all downloaded images and save with appended tag
        """
        data = isic_dataset.load_json()
        for n in tqdm(range(0,len(data))):
            img_path = op.join(data[n]['fileLoc'],data[n]['filename'])
            if op.isfile(img_path+'.jpg'):
                try:
                    im = cv2.imread(img_path+'.jpg', cv2.IMREAD_ANYCOLOR)
                    proc_image = function(im)
                    cv2.imwrite(img_path + tag +'.png',proc_image )
                    #cv2.imwrite(img_path + tag +'_32.png', imageSet.resize_to_32(proc_image))
                except cv2.error as e:
                    print(data[n]['id'])
            else:
                print(data[n]['id'])

    def resize_to_32(image):
        """Convert any size image to a 32x32 image 
        """
        try:
            return cv2.resize(image,(32,32), interpolation = cv2.INTER_CUBIC)
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
        """Convert to DCT of RGB channels
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
        """Convert to DCT of RGB channels
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

    def to_DCT_8block(image):
        """Convert to DCT of RGB channels
        """
        octave.eval('pkg load all')
        octave.addpath('m_code')
        try:
            dct8im = octave.block_dct(image,64)
            return dct8im

        except cv2.error as e:
            print(e)

    def to_DCT(image):
        """Convert to DCT of RGB channels
        """
        dctim = np.zeros(np.shape(image))
        try:
            dctim[:,:,0] = np.uint8(cv2.dct(np.float32(image[:,:,0])/255.0))*255.0    
            dctim[:,:,1] = np.uint8(cv2.dct(np.float32(image[:,:,1])/255.0))*255.0    
            dctim[:,:,2] = np.uint8(cv2.dct(np.float32(image[:,:,2])/255.0))*255.0    
            return dctim

        except cv2.error as e:
            print(e)

    def to_DCT_of_HSV(image):
        """Convert to DCT of RGB channels
        """
        image = imageSet.to_HSV(image)
        dctim = np.zeros(np.shape(image))
        try:
            dctim[:,:,0] = np.uint8(cv2.dct(np.float32(image[:,:,0])/255.0))*255.0    
            dctim[:,:,1] = np.uint8(cv2.dct(np.float32(image[:,:,1])/255.0))*255.0    
            dctim[:,:,2] = np.uint8(cv2.dct(np.float32(image[:,:,2])/255.0))*255.0    
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
    def pickle(im, b_m):
        """Label + serialized image
        """
        label = {'benign':0, 'malignant':1}
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

#   def make_binary(list_of_ids):
#       """Generates a CIFAR-like binary from a list of IDs
#       """
#       data = workingDirectory.load_json()
#       for n in list_of_ids:



