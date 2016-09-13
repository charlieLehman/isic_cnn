import json
import requests
from pprint import pprint
from tqdm import tqdm

isic_url = 'https://isic-archive.com:443/api/v1/image'
isic_payload = {'limit':'20000','offset':'0'}
isic_request = requests.get(isic_url, params=isic_payload)
img_id_list = isic_request.json()
img_meta ={} 
img_dataset = []
   
print('Accessing isic-archive\'s API. This will take a while...')
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

f = open('img_dataset.json', 'w')
f.write(json.dumps(img_dataset))
f.close()
