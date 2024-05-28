import random
import yaml
import numpy as np
import os
import re
import glob


import torch


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_nested_folder(path):
    os.makedirs(path, exist_ok=True)

def exception_not_defined(module):
    raise Exception('The '+str(module) + ' is not defined, try give definition or chech the input')


def delete_file_with_head(directory, head):
    # delete files start with head in directory
    files = glob.glob(os.path.join(directory, head+'*'))    
    for file in files:
        os.remove(file)

def find_latest_model(directory, head):
    # find the latest model saved in folder with head
    pattern = os.path.join(directory, f"{head}*epoch*.pth")
    files = glob.glob(pattern)
    max_epoch = -1
    max_file = None
    pattern = re.compile(r'epoch(\d+)\.pth$')
    for file in files:
        match = pattern.search(file)
    if match:
        epoch_num = int(match.group(1))
        if epoch_num > max_epoch:
            max_epoch = epoch_num
            max_file = file
    return file