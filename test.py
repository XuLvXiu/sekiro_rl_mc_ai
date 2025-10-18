#encoding=utf8

'''
main
'''
print('importing...')

import time
import sys
import signal
import cv2

from log import log
from pynput.keyboard import Listener, Key
import os
import numpy as np
from storage import Storage
from env import Env
import pickle
import json

g_episode_is_running = False
def signal_handler(sig, frame):
    log.debug("Gracefully exiting...")
    env.stop()
    sys.exit(0)

def on_press(key):
    # print('on_press: %s' % (key))
    global g_episode_is_running
    try:
        if key == Key.backspace: 
            log.info('The user presses backspace in the game, will terminate.')
            env.stop()
            os._exit(0)

        if hasattr(key, 'char') and key.char == ']': 
            # switch the switch
            if g_episode_is_running: 
                g_episode_is_running = False
                env.stop()
            else: 
                g_episode_is_running = True

    except Exception as e:
        print(e)


signal.signal(signal.SIGINT, signal_handler)
keyboard_listener = Listener(on_press=on_press)
keyboard_listener.start()


# create game env
env = Env()
env.eval()
env.reset()
state = env.get_state()

arr_action_id = [env.STAND_UP_ACTION_ID
]
index = 0
while True: 
    # log.info('predict main loop running')
    if not g_episode_is_running: 
        print('if you lock the boss already, press ] to begin the episode')
        time.sleep(1.0)
        env.reset()
        state = env.get_state()
        continue

    t1 = time.time()

    action_id = arr_action_id[index]
    index +=1
    index = index % len(arr_action_id)

    # do next step, get next state
    next_state, reward, is_done = env.step(action_id)
    t2 = time.time()
    log.info('predict main loop end one epoch, time: %.2f s' % (t2-t1))

    # prepare for next loop
    state = next_state

    time.sleep(1)

    '''
    if is_done: 
        env.stop()
        log.info('done.')
        break
    '''

# end of while loop

