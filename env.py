#encoding=utf8

'''
game environment
'''

import grabscreen
import window
from window import BaseWindow, global_enemy_window, player_hp_window, boss_hp_window
from utils import change_window
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from log import log
from actions import ActionExecutor
import cv2
import sys
import time
import signal
from pynput.keyboard import Listener, Key
import numpy as np
import os
from cluster_model import ClusterModel
from game_status_window import GameStatus, GameStatusWindow

class Env(object): 

    def __init__(self): 
        log.info('init env')
        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.executor = ActionExecutor('./config/actions_conf.yaml')
        # self.template_death = cv2.imread('./assets/death_crop.png', cv2.IMREAD_GRAYSCALE)

        # currently do not support JUMP
        self.arr_action_name = ['IDLE', 'ATTACK', 'PARRY', 'SHIPO', 
            # 4
            'DIANBU_ATTACK', 
            # 5
            'DOUBLE_ATTACK', 
            # 6
            'DIANBU_PARRY_ATTACK', 
            # 7
            'STAND_UP',
            # 8
            'TAKE_HULU',
            'JUMP'
        ]
        self.action_space = len(self.arr_action_name) - 1

        self.IDLE_ACTION_ID                 = 0
        self.ATTACK_ACTION_ID               = 1
        self.PARRY_ACTION_ID                = 2
        self.SHIPO_ACTION_ID                = 3
        self.DIANBU_ATTACK_ACTION_ID        = 4
        self.DOUBLE_ATTACK_ACTION_ID        = 5 
        self.DIANBU_PARRY_ATTACK_ACTION_ID  = 6
        self.STAND_UP_ACTION_ID             = 7
        self.TAKE_HULU_ACTION_ID            = 8

        self.MODE_TRAIN = 'MODE_TRAIN'
        self.MODE_EVAL  = 'MODE_EVAL'
        self.mode = None

        self.previous_action_id = -1
        self.previous_player_hp = 100
        self.previous_boss_hp   = 100
        self.is_boss_dead = False
        self.is_player_dead = False
        self.player_life = 2

        self.model = None
        self.cluster_model = None

        self.HULU_THRESHOLD = 60

        # Initialize camera
        grabscreen.init_camera(target_fps=12)
        # active and move window to top-left
        change_window.correction_window()

        if change_window.check_window_resolution_same(window.game_width, window.game_height) == False:
            raise ValueError(
                f"游戏分辨率和配置game_width({window.game_width}), game_height({window.game_height})不一致，请到window.py中修改"
            )

        if not self.wait_for_game_window_and_model(): 
            log.debug("Failed to detect game window.")
            raise ValueError('...')

        self.game_status = GameStatus()
        self.game_status_window = None

    
    def create_game_status_window(self): 
        '''
        create a tiny game status window if you like.
        '''
        self.game_status_window = GameStatusWindow(self.game_status)


    def update_game_status_window(self): 
        '''
        update game status window
        '''
        if self.game_status_window is not None: 
            self.game_status_window.update()


    def wait_for_game_window_and_model(self): 
        '''
        set window offset
        wait for model to load
        '''
        # self.model only support [IDLE, ATTACK, PARRY, SHIPO]
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_classes = 4
        print('num_classes:', num_classes)

        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, num_classes)

        model_file_name = 'model.resnet.v1'
        self.model.load_state_dict(torch.load(model_file_name))
        self.model.eval()

        if torch.cuda.is_available(): 
            self.model = self.model.cuda()

        self.cluster_model = ClusterModel()
        self.cluster_model.load()

        while True: 
            frame = grabscreen.grab_screen()
            if frame is not None and window.set_windows_offset(frame):
                log.debug("Game window detected and offsets set!")

                BaseWindow.set_frame(frame)
                BaseWindow.update_all()

                log.debug('waiting for classifier model loading...')
                image = global_enemy_window.color.copy()
                state = {
                    'image': image,
                    'player_hp': 100,
                    'boss_hp': 100,
                    'is_player_hp_down': False,
                    'is_boss_hp_down': False,
                    'HULU_THRESHOLD': self.HULU_THRESHOLD,
                }
                inputs = self.transform_state(state)

                with torch.no_grad(): 
                    if torch.cuda.is_available(): 
                        inputs = inputs.cuda()

                    outputs = self.model(inputs)

                log.debug('waiting for cluster model loading...')
                self.cluster_model.predict_env_inputs(inputs, state)

                return True
            time.sleep(1)

        return False


    def transform_state(self, state): 
        '''
        transform state:
            BGR -> RGB tensor
            add a new axis
        '''
        image = cv2.cvtColor(state['image'], cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        pil_image = self.eval_transform(pil_image)
        inputs = pil_image.unsqueeze(0)
        return inputs


    def on_action_finished(self):
        '''
        callback function of take_action
        '''
        log.debug("action execute finished")


    def is_parry(self, action_id): 
        '''
        check if action_id is parry related.
        '''
        if action_id == self.PARRY_ACTION_ID: 
            return True

        if action_id == self.STAND_UP_ACTION_ID: 
            return True

        return False


    def is_attack(self, action_id): 
        '''
        check if action is attack
        '''
        if action_id == self.ATTACK_ACTION_ID: 
            return True

        if action_id == self.DIANBU_ATTACK_ACTION_ID: 
            return True

        if action_id == self.DOUBLE_ATTACK_ACTION_ID: 
            return True

        if action_id == self.DIANBU_PARRY_ATTACK_ACTION_ID: 
            return True

        return False


    def is_take_hulu(self, action_id): 
        if action_id == self.TAKE_HULU_ACTION_ID: 
            return True

        return False


    def is_shipo(self, action_id): 
        if action_id == self.SHIPO_ACTION_ID: 
            return True

        return False


    def is_dianbu(self, action_id): 
        if action_id == self.SHIPO_ACTION_ID: 
            return True

        '''
        if action_id == self.DIANBU_ATTACK_ACTION_ID: 
            return True

        if action_id == self.DIANBU_PARRY_ATTACK_ACTION_ID: 
            return True

        if action_id == self.STAND_UP_ACTION_ID: 
            return True

        if action_id == self.TAKE_HULU_ACTION_ID: 
            return True
        '''

        return False


    def take_action(self, action_id): 
        '''
        take a new action by action_id
        wait for the action to finish
        '''
        # no idle
        # if you want to idle, please parry.
        if action_id == self.IDLE_ACTION_ID: 
            action_id = self.PARRY_ACTION_ID

        # get action name
        action_name = self.arr_action_name[action_id]

        '''
        如果现在要防御, 那么需要判断前一个动作是否为防御，
            如果前一个动作也为防御，则 IDLE 即可，因为此时还未释放右键
        如果现在不要防御，那么也需要判断前一个动作是否为防御，
            如果前一个动作为防御，那么需要释放右键才能进行本次的动作。
        '''
        if self.is_parry(action_id) and self.is_parry(self.previous_action_id): 
            action_name = 'IDLE'
        if (not self.is_parry(action_id)) and self.is_parry(self.previous_action_id): 
            log.debug('take_action: %s' % ('RELEASE_PARRY'))
            self.executor.take_action('RELEASE_PARRY', action_finished_callback=self.on_action_finished)

        log.debug('take_action: %s' % (action_name))
        self.executor.take_action(action_name, action_finished_callback=self.on_action_finished)
        while self.executor.is_running(): 
            time.sleep(0.05)

        self.previous_action_id = action_id

        # wait for boss damage to take effect.
        # how about player damage taken?
        if self.is_attack(action_id): 
            time.sleep(0.8)

        if self.is_shipo(action_id): 
            time.sleep(0.2)


    def check_done(self, state): 
        '''
        check if the player dead or if the boss dead(laugh).
        '''

        if self.mode is None: 
            log.error('please set env.mode to train() to eval()')
            sys.exit(-1)

        if self.mode == self.MODE_TRAIN: 
            # if you want to train TAKE_HULU, pls decrease this value to 15.
            if state['player_hp'] < 50: 
                self.is_player_dead = True
                self.player_life -= 1
                return True

            if state['boss_hp'] < 80: 
                self.is_boss_dead = True
                return True

        if self.mode == self.MODE_EVAL: 
            if state['player_hp'] < 1: 
                self.is_player_dead = True
                self.player_life -= 1
                return True

            if state['boss_hp'] < 1: 
                self.is_boss_dead = True
                return True

        return False


    def eval(self): 
        '''
        set mode
        '''
        self.mode = self.MODE_EVAL
        self.game_status.mode = 'EVAL'
        log.info('set new mode: %s' % (self.mode))


    def train(self): 
        '''
        set mode
        '''
        self.mode = self.MODE_TRAIN
        self.game_status.mode = 'TRAIN'
        log.info('set new mode: %s' % (self.mode))


    def reset(self): 
        '''
        reset the env
        '''
        self.previous_action_id = -1
        self.previous_player_hp = 100
        self.previous_boss_hp   = 100
        self.is_boss_dead       = False
        self.is_player_dead     = False
        self.player_life        = 2

        self.executor.interrupt_action()


    def stop(self): 
        '''
        terminate the env
        '''
        self.executor.interrupt_action()


    def get_state(self): 
        '''
        get a new state from env.
        calcaute player's hp and boss' hp
        '''
        frame = grabscreen.grab_screen()
        BaseWindow.set_frame(frame)
        BaseWindow.update_all()

        # original image, BGR format
        image = global_enemy_window.color.copy()
        player_hp = player_hp_window.get_status()
        boss_hp = boss_hp_window.get_status()

        state = {
            'image': image,
            'player_hp': player_hp,
            'boss_hp': boss_hp,
            'is_player_hp_down': False,
            'is_boss_hp_down': False,
            'HULU_THRESHOLD': self.HULU_THRESHOLD,
        }

        player_hp_down = self.previous_player_hp - player_hp 
        boss_hp_down = self.previous_boss_hp - boss_hp
        THRESHOLD = 3
        if player_hp_down > THRESHOLD: 
            state['is_player_hp_down'] = True

        if boss_hp_down > THRESHOLD: 
            state['is_boss_hp_down'] = True


        inputs = self.transform_state(state)
        state['cluster_class'] = self.cluster_model.predict_env_inputs(inputs, state)

        log.debug('get new state, hp: %5.2f %5.2f, cluster_class: %s' % (state['player_hp'], state['boss_hp'], state['cluster_class']))

        # update game status
        self.game_status.update_by_state(state)
        self.update_game_status_window()

        return state


    def step(self, action_id): 
        '''
        take a step in the env
        get new state and calculate reward
        '''

        log.debug('new step begin, action_id: %s' % (action_id))
        self.game_status.action_name = self.arr_action_name[action_id]
        self.update_game_status_window()

        self.take_action(action_id)

        new_state = self.get_state()

        is_done = self.check_done(new_state)
        (reward, log_reward) = self.cal_reward(new_state, action_id)

        log.debug('new step end, hp[%s][%s] is_done[%s], is_dead[%s][%s], player_life[%s], reward[%s %s]' % (new_state['player_hp'], new_state['boss_hp'], 
            is_done, self.is_player_dead, self.is_boss_dead,
            self.player_life,
            reward, log_reward))

        return (new_state, reward, is_done)


    def cal_reward(self, new_state, action_id): 
        '''
        calculate the reward according to the action take and the new state

        打法: 立足防御，找机会偷一刀，尽量少垫步，绝不贪刀，稳扎稳打。
        '''

        reward = 10
        log_reward = '.'

        player_hp = new_state['player_hp']
        boss_hp = new_state['boss_hp']

        player_hp_down = self.previous_player_hp - player_hp 
        boss_hp_down = self.previous_boss_hp - boss_hp
        THRESHOLD = 3

        log_reward += 'action:%s,' % (action_id)
        
        if self.is_take_hulu(action_id): 
            if self.previous_player_hp > self.HULU_THRESHOLD: 
                reward -= 50
                log_reward += 'hulu&player_hp>threshold,'

        if player_hp - self.previous_player_hp > THRESHOLD: 
            reward += 50
            log_reward += 'player_hp+,'
            if not self.is_take_hulu(action_id): 
                # reward -= 100
                log.error('error: player_hp+, but player is NOT TAKE_HULU [current hp:%s][previous:%s]' % (player_hp, self.previous_player_hp))
                self.game_status.error = 'delayed player_hp+'
                self.update_game_status_window()
                # sys.exit(-1)
        else: 
            if self.is_take_hulu(action_id): 
                reward -= 50
                log_reward += 'hulu_failed,'

        if self.is_shipo(action_id): 
            reward -= 10

        if player_hp_down > THRESHOLD: 
            # the damage maybe caused by previous actions?
            # but the previous action and the current action could become a combo.
            reward -= 20
            if self.is_attack(action_id): 
                reward -= 30
                log_reward += 'is_attack,'
            if self.is_dianbu(action_id): 
                reward -= 50
                log_reward += 'is_dianbu,'

            log_reward += 'player_hp-,'

        if boss_hp_down > THRESHOLD: 
            reward += 30
            log_reward += 'boss_hp-,'
            if not self.is_attack(action_id): 
                log.error('error: boss_hp-, but player is NOT attack')
                self.game_status.error = 'delayed boss_hp-'
                self.update_game_status_window()
        else: 
            if self.is_attack(action_id): 
                # even the boss-hp is not changed, player can interrupt boss-combo.
                reward -= 10
                reward += 0
                log_reward += 'boss_hp=,'

        self.previous_player_hp = player_hp
        self.previous_boss_hp = boss_hp

        return (reward, log_reward)


    def go_to_next_episode(self): 
        '''
        close current episode, go to the next one.
        '''
        # wait for player death
        log.info('go_to_next_episode')
        time.sleep(60)
        log.info('confirm player death')
        log.debug('take_action: NEXT_EPISODE')
        self.executor.take_action('NEXT_EPISODE', action_finished_callback=self.on_action_finished)
        while self.executor.is_running(): 
            time.sleep(0.05)

    

if __name__ == '__main__': 
    global_is_running = False
    def signal_handler(sig, frame):
        log.debug("Gracefully exiting...")
        env.stop()
        sys.exit(0)

    def on_press(key):
        print('on_press: %s' % (key))
        global global_is_running
        try:
            if key == Key.backspace: 
                log.info('The user presses backspace in the game, will terminate.')
                env.stop()
                os._exit(0)

            if hasattr(key, 'char') and key.char == ']': 
                # switch the switch
                if global_is_running: 
                    global_is_running = False
                    env.stop()
                else: 
                    global_is_running = True

        except Exception as e:
            print(e)

    signal.signal(signal.SIGINT, signal_handler)
    keyboard_listener = Listener(on_press=on_press)
    keyboard_listener.start()

    env = Env()
    # env.train()
    env.eval()
    env.reset()
    state = env.get_state()
    while True: 
        log.info('main loop running')
        if not global_is_running: 
            time.sleep(1.0)
            env.reset()
            state = env.get_state()
            continue

        t1 = time.time()

        # get action from state
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
        log.info('main loop end one epoch, time: %.2f s' % (t2-t1))

        # prepare for next loop
        state = next_state

        if is_done: 
            log.info('done.')
            break

    # end of while loop

    env.stop()
