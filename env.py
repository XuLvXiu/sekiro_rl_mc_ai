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
        self.arr_action_name = ['IDLE', 'ATTACK', 'PARRY', 'SHIPO', 'JUMP']
        self.action_space = len(self.arr_action_name) - 1

        self.IDLE_ACTION_ID     = 0
        self.ATTACK_ACTION_ID   = 1
        self.PARRY_ACTION_ID    = 2

        self.previous_action_id = -1
        self.previous_player_hp = 100
        self.previous_boss_hp   = 100

        self.model = None

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


    def wait_for_game_window_and_model(self): 
        '''
        set window offset
        wait for model to load
        '''
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_classes = self.action_space
        print('num_classes:', num_classes)

        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, num_classes)

        model_file_name = 'model.resnet.v1'
        self.model.load_state_dict(torch.load(model_file_name))
        self.model.eval()

        if torch.cuda.is_available(): 
            self.model = self.model.cuda()

        while True: 
            frame = grabscreen.grab_screen()
            if frame is not None and window.set_windows_offset(frame):
                log.debug("Game window detected and offsets set!")

                BaseWindow.set_frame(frame)
                BaseWindow.update_all()

                log.debug('waiting for model loading...')
                image = global_enemy_window.color.copy()
                state = {
                    'image': image,
                }
                inputs = self.transform_state(state)

                with torch.no_grad(): 
                    if torch.cuda.is_available(): 
                        inputs = inputs.cuda()

                    outputs = self.model(inputs)

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
        check if action_id is PARRY_ACTION_ID
        '''
        if action_id == self.PARRY_ACTION_ID: 
            return True

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
        if action_id == self.ATTACK_ACTION_ID: 
            time.sleep(0.5)


    def check_done(self, state): 
        '''
        check if the player dead or if the boss dead(laugh).
        '''
        '''
        x_min, x_max = 0, 300
        y_min, y_max = 0, 300 
        image = state['image'][y_min:y_max, x_min:x_max]
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray_image, self.template_death, cv2.TM_CCOEFF_NORMED)
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        # top_left = max_loc 

        # print('todo: check res: ', res)

        if np.max(res) >= 0.8: 
            return True

        return False
        '''

        if state['player_hp'] < 10: 
            return True

        if state['boss_hp'] < 50: 
            return True

        return False


    def reset(self): 
        '''
        reset the env
        '''
        self.previous_action_id = -1
        self.previous_player_hp = 100
        self.previous_boss_hp   = 100
        self.executor.interrupt_action()


    def stop(self): 
        '''
        stop the env
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
            'image_player_hp': player_hp_window.gray.copy(),
            'image_boss_hp': boss_hp_window.gray.copy(),
            'player_hp': player_hp,
            'boss_hp': boss_hp,
        }

        log.debug('get new state, hp: %s %s' % (player_hp, boss_hp))

        return state


    def step(self, action_id): 
        '''
        take a step in the env
        get new state and calculate reward
        '''

        log.debug('new step begin, action_id: %s' % (action_id))
        self.take_action(action_id)

        new_state = self.get_state()

        is_done = self.check_done(new_state)
        (reward, log_reward) = self.cal_reward(new_state, action_id)

        log.debug('new step end, hp[%s][%s] is_done[%s], reward[%s %s]' % (new_state['player_hp'], new_state['boss_hp'], 
            is_done, reward, log_reward))

        return (new_state, reward, is_done)


    def cal_reward(self, new_state, action_id): 
        '''
        calculate the reward according to the action take and the new state

        打法: 立足防御，找机会偷一刀，尽量少垫步，绝不贪刀，稳扎稳打。
        '''

        reward = 0
        log_reward = '.'

        player_hp = new_state['player_hp']
        boss_hp = new_state['boss_hp']

        if self.previous_player_hp - player_hp > 10: 
            # reward -= 30
            # the damage maybe caused by previous actions?
            reward -= 1
            log_reward += 'player_hp-, '

        if self.previous_boss_hp - boss_hp > 3: 
            reward += 20
            log_reward += 'boss_hp-, '

        self.previous_player_hp = player_hp
        self.previous_boss_hp = boss_hp

        return (reward, log_reward)


if __name__ == '__main__': 
    global_is_running = False
    def signal_handler(sig, frame):
        log.debug("Gracefully exiting...")
        env.stop()
        sys.exit(0)

    def on_press(key):
        print('on_press', key)
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
    state = env.get_state()
    while True: 
        log.info('main loop running')
        if not global_is_running: 
            time.sleep(1.0)
            state = env.get_state()
            env.previous_action_id = -1
            continue

        t1 = time.time()
        inputs = env.transform_state(state)

        with torch.no_grad(): 
            if torch.cuda.is_available(): 
                inputs = inputs.cuda()
            outputs = env.model(inputs)
            _, predicted = torch.max(outputs, 1)
            action_id = predicted.item()

        state, reward, is_done = env.step(action_id)
        t2 = time.time()
        log.info('main loop end one epoch, time: %.2f s' % (t2-t1))

        if is_done: 
            log.info('done.')
            break

    # end of while loop

    env.stop()
