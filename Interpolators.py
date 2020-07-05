"""
    Copyright (c) 2019 Olga Mula
"""

import numpy as np

# Interpolator
###############################################################

class Interpolator():
    def __init__(self, t_interval, data):
        """
        Assumes data is ******a list of np.array's******
        """
        
        if isinstance(data[0][1], np.ndarray) == False:
            print('P1Interp: incorrect format => reformating data...')
            for (i,pair) in enumerate(data):
                data[i] = [pair[0], np.array(pair[1], ndmin=1)]
                
        
        # reorder data 
        if data[0][0] < data[-1][0]:
            self.t_vec = [pairs[0] for pairs in data]
            self.x_vec = [pairs[1] for pairs in data]
        else:
            self.t_vec = [pairs[0] for pairs in reversed(data)]
            self.x_vec = [pairs[1] for pairs in reversed(data)]        

        self.ndim = data[0][1].ndim
        
        if self.ndim == 1:
            self.n = data[0][1].size
            self.m = None
        elif self.ndim == 2:
            self.n, self.m = data[0][1].shape
        else:
            print('Trying to interpolate a Tensor? Aborted.')
            exit('???')


class P1Interp(Interpolator):
    def __init__(self, t_interval, data):
        super().__init__(t_interval, data)
                           
    def __call__(self, t):
        
        val = None
        if self.ndim == 1:
            val = np.zeros(self.n)
            for i in range(self.n):
                    val[i] = np.interp(t, self.t_vec, [m[i].squeeze() for m in self.x_vec])
        elif self.ndim ==2:
            val = np.zeros((self.n, self.m))
            for i in range(self.n):
                for j in range(self.m):
                    val[i,j] = np.interp(t, self.t_vec, [m[i,j].squeeze() for m in self.x_vec])            
        return val
