import numpy as np


class Mesh(): # args: time bound
    
    def __init__(self, t0, T, n_points=10):
        self.t0, self.T = t0, T
        self.n = n_points
        self.h = (self.T - self.t0)/(self.n - 1)
        
        # uniform
        self.points = np.linspace(t0, T, self.n)
        
        self.intervals = list()
        self.buildIntervals()        

    def find_idx(self, t):
        it = 0
        if t <= self.t0:
            it = 0
        elif t >= self.T:
            it = self.n - 1
        else:
            it = np.max(np.where(t >= self.points)[0])
        return it

        
    def buildIntervals(self): # builds intervals based on distribution of self.points
        assert self.n > 0        
        for i in range(self.n - 1):
            self.intervals.append([self.points[i], self.points[i+1]])

    def refine(self, t): # refine locally around t
        assert t >= self.t0 and t <= self.T
        assert t not in self.points

        # Add points to grid points
        self.points = np.append(self.points, np.array(t))
        self.n += 1
        self.points.sort()

        # Rebuild intervals
        self.intervals = list()
        self.buildIntervals()

    def dyadic_refine(self):
        
        #self.n += self.n - 1                       # add mid points
        #self.points = np.linspace(self.t0, self.T, self.n)   # generate points
        
        old_mesh = self.points
        step = (old_mesh[0] + old_mesh[1]) * 0.5
        self.points = np.sort(np.concatenate(( old_mesh, old_mesh[:-1] + step )))
        self.n = np.size(self.points)
        
        # init and rebuild intervals
        self.intervals = list()
        self.buildIntervals()


    def dyadic_random_refine(self):

        assert self.n > 1
        
        r_sel = np.random.randint(self.n - 1)
        I = self.intervals[r_sel]
        mid_of_I = (I[0] + I[1]) / 2
        self.refine(mid_of_I)
