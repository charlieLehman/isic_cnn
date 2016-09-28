#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates a JSON file of data from the ISIC Archive.  
"""
import utils 


path = utils.workingDirectory.init()

utils.isic_api.get_id_list(path)
utils.isic_api.get_images(path)
