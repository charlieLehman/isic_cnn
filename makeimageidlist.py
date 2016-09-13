import json

from pprint import pprint

with open('/home/charlie/Pictures/isic_archive/image_list.json') as data_file:
    data = json.load(data_file)

f = open('test','w')
for n in range(0,len(data)):
    f.write(data[n]['_id']+'\n')
