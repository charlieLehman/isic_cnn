#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates a JSON file of data from the ISIC Archive.  
"""
import isic_api as api


path = api.input_path()

api.get_data(path)

api.get_images(path)
