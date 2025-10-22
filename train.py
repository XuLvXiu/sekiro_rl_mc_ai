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
        self.env.train()

        # number of actions.
        self.action_space = self.env.action_space

        # MC paramaters
        self.Q = Storage(self.action_space)
        self.N = Storage(self.action_space)
        self.GAMMA = 0.85

        # episode parameters
        self.MAX_EPISODES = 1000
        self.next_episode = 0
        self.CHECKPOINT_FILE = 'checkpoint.pkl'
        self.JSON_FILE = 'checkpoint.json'

        log.info('is_resume: %s' % (is_resume))
        if is_resume: 
            obj_information = self.load_checkpoint()
            self.next_episode = obj_information['episode']

        self.env.create_game_status_window()


    def train(self): 
        '''
        mc control
        '''
        begin_i = self.next_episode
        for i in range(begin_i, self.MAX_EPISODES): 
            # decay
            epsilon = 1.0 / ((i+1) / self.MAX_EPISODES + 1)
            log.info('episode: %s, epsilon: %s' % (i, epsilon))
            self.env.game_status.episode = i

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
                self.env.game_status.is_ai = False
                self.env.update_game_status_window()
                time.sleep(1.0)
                env.reset()
                state = env.get_state()
                continue

            self.env.game_status.is_ai = True

            t1 = time.time()
            log.info('generate_episode step_i: %s,' % (step_i))

            self.env.game_status.step_i     = step_i
            self.env.game_status.error      = ''
            self.env.game_status.state_id   = self.Q.convert_state_to_key(state)
            self.env.update_game_status_window()

            # get action by state
            if self.Q.has(state): 
                log.info('state[%s] found, using epsilon-greedy' % (self.Q.convert_state_to_key(state)))
                obj_found_count['y'] += 1
                Q_s = self.Q.get(state)
                probs = self.get_probs(Q_s, epsilon)
                log.info('Q_s: %s' % (Q_s))
                log.info('probs: %s' % (probs))
                action_id = np.random.choice(self.action_space, p=probs)

                '''
                # only train some states
                ########################################
                if self.Q.convert_state_to_key(state) < 7: 
                    action_id = np.argmax(Q_s)
                ########################################
                '''
            else: 
                log.info('state[%s] not found, using base-model' % (self.Q.convert_state_to_key(state)))
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
            # S0, A0, R1
            # ...   ...   ...
            # S T-1, A T-1, R T
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
        self.env.update_game_status_window()

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

            # old Q(s) and N(s)
            Q_s = self.Q.get(state)
            N_s = self.N.get(state)

            # a
            action_id = arr_action[i]

            # old Q(s, a) and old N(s, a)
            # the variable names are inappropriate.
            old_Q = Q_s[action_id]
            old_N = N_s[action_id]

            # G = the return that follows the [first] occurrence of s,a
            # arr_reward[i] is R t+1
            # arr_discount[0] is 1.0
            # arr_discount[1] is GAMMA
            # arr_discount[2] is GAMMA * GAMMA
            # G = GAMMA * G + R t+1
            arr_reward_following = arr_reward[i:] * arr_discount[:-(1+i)]
            log.info('arr_reward_following: %s' % (arr_reward_following))
            G = sum(arr_reward_following)

            # add G to Returns(s, a)
            # Q(s, a) = average(Returns(s, a))
            avg = old_Q + (G - old_Q)/(old_N+1)
            cnt = old_N + 1
            self.Q.set(state, action_id, avg)
            self.N.set(state, action_id, cnt)

            # new Q(s) and new N(s)
            # the variable names are inappropriate.
            new_Q = self.Q.get(state)
            new_N = self.N.get(state)
            log.debug('update_Q step_i: %s, old_Q[%s] old_N[%s] state[%s] action[%s] reward[%s] G[%s] new_Q[%s] new_N[%s]' % (i,
                old_Q, old_N,
                self.Q.convert_state_to_key(state),
                arr_action[i], arr_reward[i],
                G, 
                new_Q, new_N))


    def save_checkpoint(self, obj_information): 
        '''
        save checkpoint for future use.
        '''
        log.info('save_checkpoint...')
        log.info('Q: %s' % (self.Q.summary('Q')))
        log.info('N: %s' % (self.N.summary('N')))
        log.info('do NOT terminate the power, still saving...')
        log.info('actions: %s' % (self.env.arr_action_name))
        
        # pickle Q and N
        with open(self.CHECKPOINT_FILE, 'wb') as f:
            pickle.dump((self.Q, self.N), f, protocol=pickle.HIGHEST_PROTOCOL)

        log.info('still saving...')

        # write json information
        log.debug(obj_information)
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

            with open(self.JSON_FILE, 'r', encoding='utf-8') as f: 
                obj_information = json.load(f)
        except Exception as e: 
            log.error('ERROR load checkpoint: %s', (e))

        log.info('Q: %s' % (self.Q.summary('Q')))
        log.info('N: %s' % (self.N.summary('N')))
        log.debug(obj_information)
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
