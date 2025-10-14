#encoding=utf8

'''
main
'''
print('importing...')

import multiprocessing as mp
import time
import sys
import signal
import cv2

from utils import change_window
import window
from window import BaseWindow, global_enemy_window
import grabscreen
from log import log
from actions import ActionExecutor
from pynput.keyboard import Listener, Key
from pynput import mouse
import os
import numpy as np
import pandas as pd

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from storage import Storage
from train import Trainer


g_episode_is_running = False
def signal_handler(sig, frame):
    log.debug("Gracefully exiting...")
    t.stop()
    sys.exit(0)

def on_press(key):
    # print('on_press: %s' % (key))
    global g_episode_is_running
    try:
        if key == Key.backspace: 
            log.info('The user presses backspace in the game, will terminate.')
            t.stop()
            os._exit(0)

        if hasattr(key, 'char') and key.char == ']': 
            # switch the switch
            if g_episode_is_running: 
                g_episode_is_running = False
                t.stop()
            else: 
                g_episode_is_running = True

    except Exception as e:
        print(e)


signal.signal(signal.SIGINT, signal_handler)
keyboard_listener = Listener(on_press=on_press)
keyboard_listener.start()


env = Env()
action_space = env.action_space
Q = Storage(action_space)
N = Storage(action_space)
CHECKPOINT_FILE = 'checkpoint.pkl'
JSON_FILE = 'checkpoint.json'

with open(CHECKPOINT_FILE, 'rb') as f: 
    (Q, N) = pickle.load(f)
    log.info('Q.state: %s' % (self.Q.summary()))
    log.info('N.state: %s' % (self.N.summary()))

with open(self.JSON_FILE, 'r', encoding='utf-8') as f: 
    obj_information = json.load(f)

print(obj_information)

env.reset()
state = env.get_state()

while True: 
    # log.info('predict main loop running')
    if not g_episode_is_running: 
        print('if you lock the boss already, press ] to begin the episode')
        time.sleep(1.0)
        env.reset()
        state = env.get_state()
        continue

    t1 = time.time()

    # get action by state from Q
    if not self.Q.has(state): 
        log.error('why self.Q not have this state?')
        sys.exit(-1)

    Q_s = self.Q.get(state)
    a_star = np.argmax(Q_s)
    log.debug('Q_s:%s, a_star: %s' % (Q_s, a_star))
    action_id = a_star

    # do next step, get next state
    next_state, reward, is_done = env.step(action_id)
    t2 = time.time()
    log.info('predict main loop end one epoch, time: %.2f s' % (t2-t1))

    # prepare for next loop
    state = next_state

    if is_done: 
        env.stop()
        log.info('done.')
        break

# end of while loop

