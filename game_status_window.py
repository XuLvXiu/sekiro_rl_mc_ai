#encoding=utf8

import tkinter as tk

class GameStatus(): 
    '''
    game status
    '''

    def __init__(self): 
        '''
        init
        '''
        self.player_hp          = 0
        self.boss_hp            = 0
        self.action_name        = '----'
        self.is_ai              = False
        self.state_id           = -1
        self.step_i             = 0
        self.episode            = 0
        self.error              = ''
        self.mode               = ''


    def update_by_state(self, state): 
        '''
        use game state to update local variables.
        '''
        self.player_hp              = state['player_hp']
        self.boss_hp                = state['boss_hp']
        self.is_player_hp_down      = state['is_player_hp_down']
        self.is_boss_hp_down        = state['is_boss_hp_down']
        # this is next state after taking action.
        # self.state_id               = state['cluster_class']


class GameStatusWindow(): 
    '''
    a tiny tk window to show game status.
    '''

    def __init__(self, game_status): 
        '''
        init
        '''
        self.root = tk.Tk()
        self.root.title("Game Status")

        w = 300
        h = 500
        x = self.root.winfo_screenwidth() - w
        y = self.root.winfo_screenheight() - h - 100

        w = 600
        h = 250
        x = -1
        y = 720
        self.root.geometry("%dx%d+%d+%d" % (w, h, x, y))

        # frames
        self.left_frame = tk.Frame(self.root)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.right_frame = tk.Frame(self.root)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        # variables and labels
        self.variables  = {}
        self.labels     = {}

        self.add_lable('is_ai', self.left_frame)
        self.add_lable('empty', self.right_frame)

        self.add_lable('state_id', self.left_frame)
        self.add_lable('action_name', self.right_frame)

        self.add_lable('player_hp', self.left_frame)
        self.add_lable('boss_hp', self.right_frame)

        self.add_lable('episode', self.left_frame)

        self.add_lable('error', self.right_frame)

        # data source
        self.game_status = game_status


    def add_lable(self, key, frame): 
        '''
        add a new label to the frame
        '''
        self.variables[key] = tk.StringVar()
        self.labels[key] = tk.Label(frame, textvariable=self.variables[key])
        self.labels[key].config(font=('Consolas', 16))
        if key == 'is_ai' or key == 'empty': 
            self.labels[key].config(font=('Helvetica', 48))
        self.labels[key].pack(anchor="w", pady=5)


    def update(self): 
        '''
        use game_status to update local variables.
        then refresh UI.
        '''
        key = 'is_ai'
        if self.game_status.is_ai: 
            self.variables[key].set('AI')
            self.labels[key].config(fg='blue')
        else: 
            self.variables[key].set('人工')
            self.labels[key].config(fg='black')

        key = 'empty'
        self.variables[key].set('')

        key = 'action_name'
        self.variables[key].set('%s' % (self.game_status.action_name))

        key = 'state_id'
        self.variables[key].set('%s: %d' % (key, self.game_status.state_id))

        key = 'player_hp'
        self.variables[key].set('%s: %.2f' % (key, self.game_status.player_hp))
        self.labels[key].config(fg='black')
        if self.game_status.is_player_hp_down: 
            self.labels[key].config(fg='red')

        key = 'boss_hp'
        self.variables[key].set('%s: %.2f' % (key, self.game_status.boss_hp))
        self.labels[key].config(fg='black')
        if self.game_status.is_boss_hp_down: 
            self.labels[key].config(fg='red')

        key = 'episode'
        self.variables[key].set('[%s] %s: %s-%s' % (self.game_status.mode, 
            key, 
            self.game_status.episode, self.game_status.step_i))

        key = 'error'
        self.variables[key].set('')
        self.labels[key].config(fg='black')
        if len(self.game_status.error) > 0: 
            self.variables[key].set('%s: %s' % (key, self.game_status.error))
            self.labels[key].config(fg='red')

        # refresh UI
        self.root.update_idletasks()
        self.root.update()


