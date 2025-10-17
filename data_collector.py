#encoding=utf8

'''
collect image data automatically, no actions.
used for cluster model
'''
print('importing...')


from env import Env
import sys
import torch
from log import log
import cv2
import time
import signal
from pynput.keyboard import Listener, Key
import numpy as np
import os
from collections import defaultdict
import pickle
import json
import time
import os
import shutil
import argparse
from storage import Storage


class DataCollector: 
    '''
    collect data for the cluster model
    '''

    def __init__(self, is_resume=True): 
        '''
        init
        '''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info('device: %s' % (self.device))

        log.info('is_resume: %s' % (is_resume))
        if not is_resume: 
            log.info('delete old images')
            image_dir = 'images'
            if os.path.exists(image_dir): 
                shutil.rmtree(image_dir)
            os.mkdir(image_dir)

        self.env = Env()
        self.env.eval()

        # if has Q, load Q
        action_space = self.env.action_space
        self.Q = Storage(action_space)
        self.N = Storage(action_space)
        CHECKPOINT_FILE = 'checkpoint.pkl'
        JSON_FILE = 'checkpoint.json'

        if os.path.exists(CHECKPOINT_FILE) and os.path.exists(JSON_FILE): 
            log.info('loading Q')
            with open(CHECKPOINT_FILE, 'rb') as f: 
                (self.Q, self.N) = pickle.load(f)
                log.info('Q: %s' % (self.Q.summary('Q')))
                log.info('N: %s' % (self.N.summary('N')))

            with open(JSON_FILE, 'r', encoding='utf-8') as f: 
                obj_information = json.load(f)

            log.info(obj_information)
        else: 
            log.info('not using Q')


    def run(self): 
        '''
        main process
        '''
        while True: 
            time_column = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            episode = self.generate_episode()

            log.info('flush to disk, pls do NOT power off...')
            for i in range(0, len(episode)): 
                state = episode[i][0]
                image = state['image']
                cv2.imwrite('images/%s_%s.png' % (time_column, i), 
                    image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            log.info('flush done')

            self.env.go_to_next_episode()


    def generate_episode(self): 
        '''
        generate an episode use the pre-trained classifier model.
        '''
        global g_episode_is_running

        episode = []
        env = self.env

        env.reset()
        state = env.get_state()

        # g_episode_is_running = False

        step_i = 0

        while True: 
            # log.info('generate_episode main loop running')
            if not g_episode_is_running: 
                print('if you lock the boss already, press ] to begin the episode')
                time.sleep(1.0)
                env.reset()
                state = env.get_state()
                continue

            t1 = time.time()
            log.info('generate_episode step_i: %s,' % (step_i))

            # get action from state
            if self.Q.has(state): 
                log.info('state found in Q')
                Q_s = self.Q.get(state)
                a_star = np.argmax(Q_s)
                log.debug('Q_s:%s, a_star: %s' % (Q_s, a_star))
                action_id = a_star
            else: 
                log.info('state not found')
                inputs = env.transform_state(state)

                with torch.no_grad(): 
                    if torch.cuda.is_available(): 
                        inputs = inputs.cuda()
                    outputs = env.model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    action_id = predicted.item()

            # do next step, get next state
            next_state, reward, is_done = env.step(action_id)
            t2 = time.time()
            log.info('generate_episode main loop end one epoch, time: %.2f s' % (t2-t1))
            step_i += 1

            # save to current episode
            episode.append((state, action_id, reward))

            # prepare for next loop
            state = next_state

            if is_done: 
                env.stop()
                log.info('done.')
                break

        # end of while loop

        log.info('episode done. length: %s' % (len(episode)))
        return episode


    def stop(self): 
        '''
        stop the trainer
        '''
        self.env.stop()



# main

parser = argparse.ArgumentParser()
parser.add_argument('--new', action='store_true', help='new training', default=False)
args = parser.parse_args()
is_resume = not args.new

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
                # g_episode_is_running = False
                # t.stop()
                print('I cannot stop myself lalala')
            else: 
                g_episode_is_running = True

    except Exception as e:
        print(e)

signal.signal(signal.SIGINT, signal_handler)
keyboard_listener = Listener(on_press=on_press)
keyboard_listener.start()

t = DataCollector(is_resume)
t.run()
