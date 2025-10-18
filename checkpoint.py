#encoding=utf8

import time
import sys

import os
import numpy as np
from storage import Storage
import pickle
import json

# load Q
action_space = 100
Q = Storage(action_space)
N = Storage(action_space)
CHECKPOINT_FILE = 'checkpoint.pkl'
JSON_FILE = 'checkpoint.json'

with open(CHECKPOINT_FILE, 'rb') as f: 
    (Q, N) = pickle.load(f)
    print('Q: %s' % (Q.summary('Q')))
    print('N: %s' % (N.summary('N')))

with open(JSON_FILE, 'r', encoding='utf-8') as f: 
    obj_information = json.load(f)

print(obj_information)
