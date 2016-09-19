#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handles interactions with ISIC's API
"""

import sys
import json
import requests
import os.path

from tqdm import tqdm
from termcolor import colored, cprint

import utils 
import ynquery


isic_url = 'https://isic-archive.com:443/api/v1/image'

#User input to use default or custom path to store dataset
def input_path():
    if utils.state.exists():
        current_dataset = json.load(open('.state'))
        cprint("Your data is in "+ current_dataset['dir']+"\n","blue")
        img_dataset_path = current_dataset['dir']
    elif ynquery.ynQuery("Set up default path?\n /tmp/isic_dataset/"):
        img_dataset_path = '/tmp/isic_dataset/'
    else:
        dir_query = colored("Where should I set up the root directory for the dataset?\n", "blue")
        while True:
            try:
                img_dataset_path = input(dir_query)
                os.path.isdir(img_dataset_path)
                img_dataset_path = os.path.join(img_dataset_path, 'isic_dataset')
                break
            except FileNotFoundError:
                colored("Please input a valid directory.\n","yellow")

    if os.path.exists(img_dataset_path):
        if ynquery.ynQuery("It looks like you already have the dataset. "
                            "Should I overwrite? "):
            return img_dataset_path
        else:
            sys.exit(1)
    else:
        return img_dataset_path


#Generates a JSON file for the dataset
def get_data(directory):
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
        isic_request = requests.get(isic_url, params=isic_payload)
    except requests.exceptions.RequestException as e:
        cprint('There seems to be an issue with connecting...\n', 'red')
        return "Error: {}".format(e)
        sys.exit(1)

    img_id_list = isic_request.json()
    img_meta = {}
    img_dataset = []

    cprint('Accessing isic-archive\'s API. This will take a while...', 'blue')

#Iteratively query the API for each ID
    for n in tqdm(range(0, len(img_id_list))):
        img_meta_request = requests.get(isic_url + '/' + img_id_list[n]['_id'])
        assert isinstance(img_meta_request, object)
        img_meta_list = img_meta_request.json()
        img_meta['id'] = img_id_list[n]['_id']
        img_meta['filename'] = img_meta_list['name']
        img_meta['age'] = img_meta_list['meta']['clinical'].get('age')
        img_meta['sex'] = img_meta_list['meta']['clinical'].get('sex')
        img_meta['b_m'] = img_meta_list['meta']['clinical'].get('benign_malignant')
        img_meta['diagnosis'] = img_meta_list['meta']['clinical'].get('diagnosis')
        if img_meta not in img_dataset:
            img_dataset.append(dict(img_meta))

#Save the dataset to a directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    f = open(os.path.join(directory, 'img_dataset.json'), 'w')
    f.write(json.dumps(img_dataset))
    f.close()
    cprint('Dataset is in ' + os.path.abspath(directory), 'blue')
    utils.state.set(directory)

#Get the images from ISIC archive
def get_images(directory):
    while True:
        try:
            image_size = int(input(colored('What size of square image in pixels?\n', 'blue')))
            break
        except ValueError: 
            cprint('Please input an integer', 'red')
    try:
        with open(os.path.join(directory,'img_dataset.json')) as img_dataset:
            data = json.load(img_dataset)

        __save_images__(directory, image_size, 'benign', [x for x in data if x['b_m'] == 'benign'])
        __save_images__(directory, image_size, 'malignant', [x for x in data if x['b_m'] == 'malignant'])

    except FileNotFoundError as e:
        cprint('There is nothing here..\n', 'red')
        print(e)
        sys.exit(1)

#Saves the image for get_images
def __save_images__(directory, image_size, name, id_set):
    isic_payload = {'width':image_size}
    print('Getting '+ name +' images \n')
    imdir = os.path.join(directory,'images',name)
    if not os.path.exists(imdir):
        os.makedirs(imdir)
    for n in tqdm(range(0,len(id_set))):
        img_name = id_set[n]['filename']
        with open(os.path.join(imdir,img_name)+'.jpg','wb') as f:
            isic_request = requests.get(isic_url+'/'+id_set[n]['id']+'/thumbnail', params=isic_payload)
            f.write(isic_request.content)
