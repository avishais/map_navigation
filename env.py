import numpy as np
import matplotlib.pyplot as plt
from astar import Astar
from scipy.stats import rv_discrete


class Env(Astar):
    num_obs = 5
    D = {'u': np.array([-1,0]), 'd': np.array([1,0]), 'r': np.array([0,1]), 'l': np.array([0,-1]), 'f': np.array([0, 0])}
    alpha = 1.0

    def __init__(self, nrows = 9, ncols = 9, loc = [0,0]):
        super(Env, self).__init__()
        self.nrows, self.ncols = nrows, ncols
        self.num_free_cells = (ncols*nrows - self.num_obs)
        self.env = np.ones((self.nrows, self.ncols))
        self.prob_map = np.zeros((self.nrows, self.ncols)) + 1./self.num_free_cells
        self.obs_map = np.zeros((self.nrows, self.ncols))
        self.score_map = np.zeros((self.nrows, self.ncols))
        self.path = []
        self.path.append(loc)

        self.set_obs()
        self.Map = np.copy(self.obs_map)
        self.directions = self.D

    def set_start_state(self, x):
        self.path = []
        self.path.append(x)

    def set_obs(self):
        np.random.seed(1)
        obs = np.random.randint(0, high = [self.nrows, self.ncols], size=(self.num_obs, 2))

        for o in obs:
            self.env[o[0], o[1]] = 0.
            self.prob_map[o[0], o[1]] = 0
            self.obs_map[o[0], o[1]] = 1
        self.normalize_map()

        self.direc_prob = {key: 0 for key in self.D.keys()} 
        for x in range(self.ncols):
            for y in range(self.nrows):
                if self.obs_map[x,y]:
                    continue
                for d in self.D.keys():
                    if d == 'f':
                        if x + 1 < self.ncols and y + 1 < self.nrows and y - 1 >= 0 and x - 1 >= 0 and not self.obs_map[x + 1, y] and not self.obs_map[x - 1, y] and not self.obs_map[x, y - 1] and not self.obs_map[x, y + 1]:
                            self.direc_prob[d] += 1                            
                    else:
                        if x + self.D[d][0] >= self.ncols or y + self.D[d][1] >= self.nrows or self.obs_map[x + self.D[d][0], y + self.D[d][1]]:
                            self.direc_prob[d] += 1
        for d in self.direc_prob.keys():
            self.direc_prob[d] /= self.num_free_cells

        for x in range(self.ncols):
            for y in range(self.nrows):
                if self.obs_map[x,y]:
                    self.score_map[x, y] = 100
                    continue
                S = self.sense([x, y])
                self.score_map[x, y] = -len(S)*2
        # self.set_score_map(self.score_map)

    def normalize_map(self):
        self.prob_map /= np.sum(self.prob_map)

    def update_prob(self, direction = 'u'):
        # https://ae640a.github.io/assets/winter17/references/AMRobots5.pdf, page 191
        
        for x in range(self.ncols):
            for y in range(self.nrows):
                v = np.array([x,y]) + self.D[direction]
                if np.any(v < 0) or v[1] >= self.ncols or v[0] >= self.nrows or self.obs_map[v[0], v[1]] == 1:
                    self.prob_map[x, y] *= self.direc_prob[direction]
                else:
                    self.prob_map[x, y] = 0.
        self.normalize_map()

    def action(self, direction, loc = [0,0]):
        temp = np.copy(self.prob_map)

        for x in range(self.nrows):
            for y in range(self.ncols):
                v = np.array([x,y]) + self.D[direction]
                if np.all(v >= 0) and v[0] < self.nrows and v[1] < self.ncols and self.obs_map[x,y]:
                    self.prob_map[v[0],v[1]] = 0.
                    continue
                if np.all(v >= 0) and v[0] < self.nrows and v[1] < self.ncols and self.obs_map[v[0],v[1]]:
                    continue
                if np.all(v >= 0) and v[0] < self.nrows and v[1] < self.ncols:
                    self.prob_map[v[0],v[1]] = temp[x,y]

        if direction == 'u':
            self.prob_map[-1,:] = 0.
        elif direction == 'd':
            self.prob_map[0,:] = 0.
        elif direction == 'r':
            self.prob_map[:,0] = 0.
        elif direction == 'l':
            self.prob_map[:,-1] = 0.
        self.normalize_map()

        self.path.append(loc + self.D[direction])
        return loc + self.D[direction]

    def sense(self, x):
        S = []
        for d in self.D.keys():
            if d == 'f':
                if self.obs_map[x[0], x[1]]:
                    print("In collision!!!")
                    exit(1)
            else:
                if np.any(x + self.D[d] < 0) or x[0] + self.D[d][0] >= self.ncols or x[1] + self.D[d][1] >= self.nrows or self.obs_map[x[0] + self.D[d][0], x[1] + self.D[d][1]]:
                    S.append(d)
        return S

    def get_max_prob(self):
        return np.max(self.prob_map)
    
    def get_random_action(self, x):
        S = []
        for d in self.D.keys():
            if d == 'f':
                continue
            else:
                if np.all(x + self.D[d] >= 0) and x[0] + self.D[d][0] < self.ncols and x[1] + self.D[d][1] < self.nrows and not self.obs_map[x[0] + self.D[d][0], x[1] + self.D[d][1]]:
                    S.append(d)
        return S[np.random.randint(len(S))]

    def gen_random_state(self):
        np.random.seed()
        while 1:
            x = np.random.randint(low = 0, high = [self.nrows, self.ncols], size=(2,))
            if not self.obs_map[x[0], x[1]]:
                break
        np.random.seed(1)
        return x

    def g_score(self, x):
        self.alpha = np.maximum(self.get_max_prob(), 0.1)
        # print(self.score_map[x[0], x[1]], 1.0, self.alpha)
        return (1 - self.alpha) * self.score_map[x[0], x[1]] + self.alpha * 1.0

    def sample_state_distribution(self):
        np.random.seed()
        idx = np.array(range(self.nrows*self.ncols)).reshape(self.nrows, self.ncols)
        smp = rv_discrete(values=(idx, self.prob_map)).rvs(size = 1)

        coor = np.where( idx == smp[0] )
        coor = [ coor[0][0], coor[1][0] ]

        np.random.seed(1)
        return coor

    def plot(self, map = False, P = []):
        if map:
            row_labels = range(self.nrows)
            col_labels = range(self.ncols)

            path = np.array(self.path)
            for x, p in zip(path, P):
                self.env[x[0], x[1]] = 0.6*p+0.2

            plt.matshow(self.env, cmap='gray',vmin=0,vmax=1)
            for ir in row_labels:
                plt.plot([-0.5, self.ncols-0.5], [ir-0.5, ir-0.5], '-k', linewidth = 1)
            for ir in col_labels:
                plt.plot([ir-0.5, ir-0.5], [-0.5, self.nrows-0.5], '-k', linewidth = 1)

            plt.plot(path[0,1], path[0,0], 'pg')
            plt.plot(path[:,1], path[:,0], '.-r')
            plt.plot(path[0,1], path[0,0], 'pg')
            plt.plot(path[-1,1], path[-1,0], 'ob')
            plt.xticks(range(self.nrows), col_labels)
            plt.yticks(range(self.nrows), row_labels)
            plt.xlabel('y')
            plt.ylabel('x')

        plt.figure()
        # self.prob_map = np.ma.masked_where(self.prob_map < 0., self.prob_map)
        cmap = plt.cm.OrRd
        # cmap.set_under(color='black')  
        # cmap.set_bad(color='black')

        cax = plt.imshow(self.prob_map, cmap=cmap, vmin=0, vmax=np.max(self.prob_map), interpolation='nearest') #cmap=plt.cm.bone
        plt.colorbar(cax)
        plt.plot(path[:,1], path[:,0], '.-k')
        plt.plot(path[-1,1], path[-1,0], 'ok')
        plt.xlabel('y')
        plt.ylabel('x')

        plt.show()


# E = Env(20, 20)
# x = E.gen_random_state()
# E.set_start_state(x)

# sol = E.plan(x_start, x_goal)
# if not sol:
#     print("Path not found!")
#     exit(1)
# print (sol)
# path = sol['path']
# actions = sol['actions']
# E.plot_plan(path, do_plot = False)

# x = np.copy(x_start)
# for a in actions:
#     S = E.sense(x)
#     for s in S:
#         E.update_prob(s)
#     x = E.action(a, loc = x)
#     print(a, E.get_max_prob())
# E.plot(True)

# E.plot(True)

# while E.get_max_prob() < 0.95:
#     S = E.sense(x)
#     for s in S:
#         E.update_prob(s)
#     a = E.get_random_action(x)
#     x = E.action(a, loc = x)
#     print(a, E.get_max_prob())
# E.plot(True)
