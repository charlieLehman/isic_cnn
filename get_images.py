import json
import requests
from pprint import pprint
from tqdm import tqdm

isic_url = 'https://isic-archive.com:443/api/v1/image'
isic_payload = {'width':'256'}
with open('img_dataset.json') as img_dataset:
    data = json.load(img_dataset)

benign_id = [x for x in data if x['b_m'] == 'benign']
malign_id = [x for x in data if x['b_m'] == 'malignant']

print('Getting benign images \n')
for n in tqdm(range(0,len(benign_id))):
    img_name = benign_id[n]['filename']
    with open('benign/'+img_name +'.jpg','wb') as b_handle:
        isic_request = requests.get(isic_url+'/'+benign_id[n]['id']+'/thumbnail', params=isic_payload)
        b_handle.write(isic_request.content)

print('Getting malignant images \n')
for n in tqdm(range(0,len(malign_id))):
    img_name = malign_id[n]['filename']
    with open('malignant/'+img_name +'.jpg','wb') as m_handle:
        isic_request = requests.get(isic_url+'/'+malign_id[n]['id']+'/thumbnail', params=isic_payload)
        m_handle.write(isic_request.content)
