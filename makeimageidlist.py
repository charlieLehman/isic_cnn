import json
import requests
import subprocess
from pprint import pprint

img_id_url = 'https://isic-archive.com:443/api/v1/image'
img_metadata_url = 'https://isic-archive.com:443/api/v1/image/'
img_id_payload = {'limit':'2'}
img_id_request = requests.get(img_id_url, params=img_id_payload)
img_id_list = img_id_request.json()
   
for n in range(0, len(img_id_list)-1):
    #print(img_id_list[n]["_id"])#Print all ID's from get request
    img_metadata_request = requests.get(img_metadata_url+img_id_list[n]['_id'])
    img_metadata_
print(img_metadata_request.json())
