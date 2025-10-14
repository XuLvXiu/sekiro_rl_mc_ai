#encoding=utf8

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
import argparse
from storage import Storage


class Trainer: 
    '''
    train a Monte-Carlo(MC) agent.
    '''

    def __init__(self, is_resume=True): 
        '''
        init
        '''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info('device: %s' % (self.device))

        self.env = Env()

        # number of actions.
        self.action_space = self.env.action_space

        # MC paramaters
        self.Q = Storage(self.action_space)
        self.N = Storage(self.action_space)
        self.GAMMA = 0.9

        # episode parameters
        self.MAX_EPISODES = 2000
        self.next_episode = 0
        self.CHECKPOINT_FILE = 'checkpoint.pkl'
        self.JSON_FILE = 'checkpoint.json'

        log.info('is_resume: %s' % (is_resume))
        if is_resume: 
            obj_information = self.load_checkpoint()
            self.next_episode = obj_information['episode']


    def train(self): 
        '''
        mc control
        '''
        begin_i = self.next_episode
        for i in range(begin_i, self.MAX_EPISODES): 
            # decay
            epsilon = 1.0 / ((i+1) / 8000 + 1)
            log.info('episode: %s, epsilon: %s' % (i, epsilon))

            episode = self.generate_episode_from_Q(epsilon)
            self.update_Q(episode)

            self.next_episode += 1

            obj_information = {
                'episode': self.next_episode,
                'time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            }
            self.save_checkpoint(obj_information)

            self.env.go_to_next_episode()

        log.info('mission accomplished :)')
    

    def generate_episode_from_Q(self, epsilon): 
        '''
        generate an episode using epsilon-greedy policy
        '''
        # global g_episode_is_running

        episode = []
        env = self.env

        env.reset()
        state = env.get_state()

        # g_episode_is_running = False
        obj_found_count = {
            'y': 0,
            'n': 0,
        }

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

            # get action by state
            if self.Q.has(state): 
                log.info('state found, using epsilon-greedy')
                obj_found_count['y'] += 1
                Q_s = self.Q.get(state)
                probs = self.get_probs(Q_s, epsilon)
                log.info('Q_s: %s' % (Q_s))
                log.info('probs: %s' % (probs))
                action_id = np.random.choice(self.action_space, p=probs)
            else: 
                log.info('state not found, using base-model')
                obj_found_count['n'] += 1

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

        log.info('episode done. length: %s, found_count: %s' % (len(episode), 
            obj_found_count))
        return episode


    def get_probs(self, Q_s, epsilon): 
        '''
        obtain the action probabilities related to the epsilon-greedy policy.
        '''
        ones = np.ones(self.action_space)
        # default action probability
        policy_s = ones * epsilon / self.action_space

        # best action probability
        a_star = np.argmax(Q_s)
        log.info('a_star: %s' % (a_star))
        policy_s[a_star] = 1 - epsilon + epsilon / self.action_space

        return policy_s


    def update_Q(self, episode): 
        '''
        update Q using the episode
        '''
        # unzip 
        arr_state, arr_action, arr_reward = zip(*episode)
        log.debug('after unzip: %s %s %s' % (len(arr_state),
            len(arr_action), len(arr_reward)))

        length = len(arr_reward)
        arr_discount = np.array([self.GAMMA**i for i in range(length+1)])

        for i in range(0, length): 
            state = arr_state[i]
            Q_s = self.Q.get(state)
            N_s = self.N.get(state)
            action_id = arr_action[i]

            old_Q = Q_s[action_id]
            old_N = N_s[action_id]

            # G = the return that follows the [first] occurrence of s,a
            # we do not need to collect reward from so much following states.
            # so we can use GAMMA to decay the rewards.
            arr_reward_following = arr_reward[i:] * arr_discount[:-(1+i)]
            log.info('arr_reward_following: %s' % (arr_reward_following))
            G = sum(arr_reward_following)

            # Q(s, a) = average(Return(s, a))
            avg = old_Q + (G - old_Q)/(old_N+1)
            cnt = old_N + 1
            self.Q.set(state, action_id, avg)
            self.N.set(state, action_id, cnt)

            new_Q = self.Q.get(state)
            new_N = self.N.get(state)
            log.debug('update_Q step_i: %s, old_Q[%s] old_N[%s] action[%s] reward[%s] G[%s] new_Q[%s] new_N[%s]' % (i,
                old_Q, old_N,
                arr_action[i], arr_reward[i],
                G, 
                new_Q, new_N))


    def save_checkpoint(self, obj_information): 
        '''
        save checkpoint for future use.
        '''
        log.info('save_checkpoint...')
        log.info('Q.state: %s' % (self.Q.summary()))
        log.info('N.state: %s' % (self.N.summary()))
        log.info('do NOT terminiate the power, still saving...')
        
        # pickle Q and N
        with open(self.CHECKPOINT_FILE, 'wb') as f:
            pickle.dump((self.Q, self.N), f, protocol=pickle.HIGHEST_PROTOCOL)

        log.info('still saving...')

        # write json information
        with open(self.JSON_FILE, 'w', encoding='utf-8') as f: 
            json.dump(obj_information, f, indent=4, ensure_ascii=False) 

        log.info('saved ok')


    def load_checkpoint(self): 
        '''
        load history checkpoint
        '''
        log.info('load_checkpoint')
        obj_information = {'episode': 0}
        try: 
            with open(self.CHECKPOINT_FILE, 'rb') as f: 
                (self.Q, self.N) = pickle.load(f)
                log.info('Q.state: %s' % (self.Q.summary()))
                log.info('N.state: %s' % (self.N.summary()))

            with open(self.JSON_FILE, 'r', encoding='utf-8') as f: 
                obj_information = json.load(f)
        except Exception as e: 
            print(e)

        print(obj_information)
        return obj_information


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

t = Trainer(is_resume)
t.train()
