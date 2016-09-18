import json
import os.path
import shutil
import ynquery

def set_state(directory):
    f = open('.state', 'w')
    state = {}
    state['dir'] = directory
    json.dump(state,f)
    f.close()

def check_state():
    return os.path.isfile('.state')

def clean_up():
    if os.path.isfile('.state'):
        current_dataset = json.load(open('.state'))
        if ynquery.ynQuery("Should I delete " + current_dataset['dir']):
            shutil.rmtree(current_dataset['dir'])
            os.remove('.state')
