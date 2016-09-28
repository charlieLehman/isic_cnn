import json
import sys
import os.path
import shutil
import requests
from multiprocessing import Process, Pool
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
                    os.path.isdir(dataset_path)
                    dataset_path = os.path.join(dataset_path, 'isic_dataset')
                    break
                except FileNotFoundError:
                    colored("Please input a valid directory.\n","yellow")

        if os.path.exists(dataset_path):
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
        return os.path.isfile('.workingDirectory')

    def clean_up():
        if os.path.isfile('.workingDirectory'):
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
        img_meta['count_benign']=0
        img_meta['count_malign']=0

        cprint('Accessing isic-archive\'s API to retreive dataset ID list. This will take a while...', 'blue')

#Iteratively query the API for each ID
        for n in tqdm(range(0, count_images)):
            img_meta_request = requests.get(isic_api.isic_url + '/' + img_id_list[n]['_id'])
            assert isinstance(img_meta_request, object)
            img_meta_list = img_meta_request.json()
            img_meta['id'] = img_id_list[n]['_id']
            img_meta['filename'] = img_meta_list['name']
            img_meta['age'] = img_meta_list['meta']['clinical'].get('age')
            img_meta['sex'] = img_meta_list['meta']['clinical'].get('sex')
            img_meta['b_m'] = img_meta_list['meta']['clinical'].get('benign_malignant')
            img_meta['diagnosis'] = img_meta_list['meta']['clinical'].get('diagnosis')
            if img_meta not in dataset:
                dataset.append(dict(img_meta))

#Save the dataset to a directory
        if not os.path.exists(directory):
            os.makedirs(directory)
        f = open(os.path.join(directory, 'dataset.json'), 'w')
        f.write(json.dumps(dataset))
        f.close()
        cprint('Dataset is in ' + os.path.abspath(directory), 'blue')
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
            with open(os.path.join(directory,'dataset.json')) as dataset:
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
        imdir = os.path.join(directory,'images',name)
        if not os.path.exists(imdir):
            os.makedirs(imdir)
        for n in tqdm(range(0,len(id_set))):
            img_name = id_set[n]['filename']
            with open(os.path.join(imdir,img_name)+'.jpg','wb') as f:
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
class dataset:                
#Return count of entry
    def how_many(value, field):
        try:
            with open(os.path.join(workingDirectory.get(),'dataset.json')) as dataset:
                return len([x for x in json.load(dataset) if x[field] == value])

        except FileNotFoundError as e:
            cprint('There is nothing here..\n', 'red')
            print(e)
            sys.exit(1)
