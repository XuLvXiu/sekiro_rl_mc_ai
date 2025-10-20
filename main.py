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
env.create_game_status_window()
env.eval()

# load Q
action_space = env.action_space
Q = Storage(action_space)
N = Storage(action_space)
CHECKPOINT_FILE = 'checkpoint.pkl'
JSON_FILE = 'checkpoint.json'

with open(CHECKPOINT_FILE, 'rb') as f: 
    (Q, N) = pickle.load(f)
    log.info('Q: %s' % (Q.summary('Q')))
    log.info('N: %s' % (N.summary('N')))

with open(JSON_FILE, 'r', encoding='utf-8') as f: 
    obj_information = json.load(f)

log.info(obj_information)
env.game_status.episode = obj_information['episode']

'''
# generate the policy
policy = {}
for (k, v) in Q.items(): 
    state = k
    Q_s = v
    action_id = np.argmax(Q_s)
    policy[state] = action_id
log.info('policy: %s' % (policy))
'''



env.reset()
state = env.get_state()

step_i = 0
while True: 
    # log.info('predict main loop running')
    if not g_episode_is_running: 
        print('if you lock the boss already, press ] to begin the episode')
        env.game_status.is_ai = False
        env.update_game_status_window()
        time.sleep(1.0)
        env.reset()
        state = env.get_state()
        continue

    env.game_status.is_ai = True

    t1 = time.time()

    env.game_status.step_i     = step_i
    env.game_status.error      = ''
    env.game_status.state_id   = Q.convert_state_to_key(state)
    env.update_game_status_window()

    # get action by state from Q
    if not Q.has(state): 
        log.error('why Q not have this state?')
        sys.exit(-1)

    Q_s = Q.get(state)
    a_star = np.argmax(Q_s)
    log.debug('Q_s:%s, a_star: %s' % (Q_s, a_star))
    action_id = a_star

    # do next step, get next state
    next_state, reward, is_done = env.step(action_id)
    t2 = time.time()
    log.info('predict main loop end one epoch, time: %.2f s' % (t2-t1))

    step_i += 1

    # prepare for next loop
    state = next_state

    if is_done: 
        log.info('is_done(hp < 1), re-check')
        # one more double attack to kill(REN_SHA, deathblow) the BOSS or take a new life.
        if env.is_boss_dead: 
            time.sleep(0.1)
            env.take_action(env.DOUBLE_ATTACK_ACTION_ID)
            # if boss dead, stop the game.

        if env.is_player_dead and env.player_life > 0: 
            time.sleep(10)
            env.take_action(env.ATTACK_ACTION_ID)
            time.sleep(1)
            env.is_player_dead = False
            state = env.get_state()
            # if take a new life, continue the game
            continue

        env.stop()
        log.info('done.')
        break

# end of while loop
env.update_game_status_window()

