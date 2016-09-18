#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handles yes or no queries
"""
from termcolor import colored, cprint

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
