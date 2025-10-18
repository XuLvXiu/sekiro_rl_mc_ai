#encoding=utf8

'''
Storage schema for Q and N, which are dict of array: 
Q = {
    state_key_1: [0, 0, 0, 0],
    state_key_2: [0, 0, 0, 0],
    ....
}
the length of the array is action_space.
'''
from collections import defaultdict
import json
import numpy as np

class Storage: 
    '''
    storage schema for Q and N
    '''

    def __init__(self, d2_length): 
        '''
        init
        '''
        # self.obj = defaultdict(lambda: np.zeros(d2_length))
        self.obj = {}
        self.d2_length = d2_length


    def convert_state_to_key(self, state): 
        key = state['cluster_class']
        # print('len key: ', len(key))
        return key


    def has(self, state): 
        '''
        check if state in the object
        '''
        key = self.convert_state_to_key(state)
        if key in self.obj: 
            return True

        return False


    def get(self, state): 
        '''
        get the value from obj by state
        if the state not exist in the object, will return default value
        '''
        key = self.convert_state_to_key(state)
        if key not in self.obj: 
            self.obj[key] = np.zeros(self.d2_length)

        return self.obj[key]


    def set(self, state, action_id, value): 
        '''
        set new value to obj
        '''
        key = self.convert_state_to_key(state)
        if key not in self.obj: 
            self.obj[key] = np.zeros(self.d2_length)
        self.obj[key][action_id] = value


    def length(self): 
        '''
        return real length of the obj
        '''

        return len(self.obj)


    def summary(self, name): 
        '''
        get the summary of the obj
        '''
        length = len(self.obj)
        str_summary = 'length: %s' % (length)

        str_summary += '\n'
        for (k, v) in sorted(self.obj.items()): 
            if name == 'Q': 
                str_summary += 'state: %2s, argmax: %s, Q_s: %s\n' % (k, np.argmax(v), v)
            else: 
                str_summary += 'state: %2s, sum: %s, N_s: %s\n' % (k, np.sum(v), v)

        return str_summary 
    

