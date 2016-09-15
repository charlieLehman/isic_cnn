import sys
from termcolor import colored, cprint
import json
import requests
import os.path
from pprint import pprint
from tqdm import tqdm
    
img_dataset_path = '/tmp/isic_dataset/'

def overwrite_yes_no(question, default="no"):
    valid = {"yes" :True, "y": True, "ye": True, "no": False, "n": False}
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
                             "(or 'y' or 'n').\n" , "red")

def get_data():
    while True:
        try:
            payload_query = colored('How many entries should I ask for?\n','blue')
            payload_limit = int(input(payload_query))
            break
        except ValueError:
            cprint('Please input an integer', 'red')

    isic_url = 'https://isic-archive.com:443/api/v1/image'
    isic_payload = {'limit':payload_limit,'offset':'0'}

    try:
        isic_request = requests.get(isic_url, params=isic_payload)
    except requests.exceptions.RequestException as e:
        cprint('There seems to be an issue with connecting...\n','red')
        return "Error: {}".format(e)
        sys.exit(1)

    img_id_list = isic_request.json()
    img_meta ={} 
    img_dataset = []

    cprint('Accessing isic-archive\'s API. This will take a while...','blue')
    for n in tqdm(range(0, len(img_id_list))):
        img_meta_request = requests.get(isic_url+'/'+img_id_list[n]['_id'])
        img_meta_list = img_meta_request.json()
        img_meta['id'] = img_id_list[n]['_id']
        img_meta['filename'] = img_meta_list['name']
        img_meta['age'] = img_meta_list['meta']['clinical'].get('age')
        img_meta['sex'] = img_meta_list['meta']['clinical'].get('sex')
        img_meta['b_m'] = img_meta_list['meta']['clinical'].get('benign_malignant')
        img_meta['diagnosis'] = img_meta_list['meta']['clinical'].get('diagnosis')
        if img_meta not in img_dataset:
            img_dataset.append(dict(img_meta))
    f = open(img_dataset_path + 'img_dataset.json', 'w')
    f.write(json.dumps(img_dataset))
    f.close()
    cprint('Dataset is in ' + img_dataset_path, 'blue')

if os.path.exists(img_dataset_path):
    if overwrite_yes_no("It looks like you already have the dataset. "
            "Should I overwrite? "):
        get_data()
else:
    os.makedirs(img_dataset_path)
    get_data()
    
