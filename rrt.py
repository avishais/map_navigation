"""

Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)

"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np

show_animation = False


class RRT:
    """
    Class for RRT planning
    """

    directions = {'u': np.array([-1,0]), 'd': np.array([1,0]), 'r': np.array([0,1]), 'l': np.array([0,-1])}


    class Node:
        """
        RRT Node
        """

        def __init__(self, x, y, idx = None):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None
            self.parent_action = None
            self.idx = idx
        
        def coord(self):
            print("Position <%d,%d>"%(self.x,self.y))

    def __init__(self, Map,
                 expand_dis=1.0, path_resolution=0.5, goal_sample_rate=5, max_iter=10500):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        Map:obstacle Positions [[x,y,size],...]

        """
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.Map = Map
        self.node_list = []
        self.nrows = self.Map.shape[0]
        self.ncols = self.Map.shape[1]

    def plan(self, start, goal, animation=True):
        """
        rrt path planning

        animation: flag for animation on or off
        """
        self.start = self.Node(start[0], start[1], idx = 0)
        self.end = self.Node(goal[0], goal[1])
        self.node_list = []

        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node):
                self.node_list.append(new_node)
                self.node_list[-1].idx = len(self.node_list)-1

            if animation and i % 5 == 0:
                self.plot_plan(rnd_node)

            if self.node_list[-1].x == self.end.x and self.node_list[-1].y == self.end.y: 
                return self.generate_final_course(len(self.node_list) - 1)

            if animation and i % 5:
                self.plot_plan(rnd_node)

        return None  # cannot find path

    def steer(self, from_node, to_node, extend_length=1, rand = False):

        new_node = self.Node(from_node.x, from_node.y)

        if rand:
            a = random.choice(list(self.directions.keys()))
        else:
            A = []
            for a in self.directions.keys():
                x = new_node.x + self.directions[a][0]
                y = new_node.y + self.directions[a][1]

                l = (x - to_node.x) ** 2 + (y - to_node.y) ** 2
                A.append(l)
            a = list(self.directions.keys())[np.argmin(A)]

        new_node.x += self.directions[a][0]
        new_node.y += self.directions[a][1]

        new_node.parent = from_node.idx
        new_node.parent_action = a

        return new_node

    def generate_final_course(self, goal_ind):
        path = []
        action_path = []
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            action_path.append(node.parent_action)
            node = self.node_list[node.parent]
        path.append([node.x, node.y])
        action_path.append(node.parent_action)
        return {'path': path[::-1], 'actions': action_path[::-1][1:]}

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            x = np.random.randint(low = 0, high = [self.nrows, self.ncols], size=(2,))    
            rnd = self.Node(x[0], x[1])
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def plot_plan(self, rnd=None, path = None, stop = True):
        row_labels = range(self.nrows)
        col_labels = range(self.ncols)

        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])

        plt.imshow(np.logical_not(self.Map), cmap='gray',vmin=0,vmax=1)
        for ir in row_labels:
            plt.plot([-0.5, self.ncols-0.5], [ir-0.5, ir-0.5], '-k', linewidth = 1)
        for ir in col_labels:
            plt.plot([ir-0.5, ir-0.5], [-0.5, self.nrows-0.5], '-k', linewidth = 1)

        if rnd is not None:
            plt.plot(rnd.y, rnd.x, "^r")
        for node in self.node_list:
            if node.parent is not None:
                plt.plot([node.y, self.node_list[node.parent].y], [node.x, self.node_list[node.parent].x], "-g", linewidth = 3.0)

        plt.plot(self.start.y, self.start.x, 'pg')
        plt.plot(self.end.y, self.end.x, 'ob')

        plt.xticks(range(self.nrows), col_labels)
        plt.yticks(range(self.nrows), row_labels)
        plt.xlabel('y')
        plt.ylabel('x')

        if path is not None and stop:
            plt.plot([y for (x, y) in path], [x for (x, y) in path], '-r')
            plt.show()
        elif path is not None:
            plt.plot([y for (x, y) in path], [x for (x, y) in path], '-r')
            plt.pause(0.01)
        else:
            plt.pause(0.01)

    def check_collision(self, node):

        if node is None:
            return False

        if not (0 <= node.x < self.nrows and 0 <= node.y < self.ncols and self.Map[node.x][node.y] == 0):
            return False

        return True  # safe

    def get_random_action(self, theta):
        b = np.pi/4.0
        if -b < theta <= b:
            return 'r'
        elif b < theta <= 3*b:
            return 'u'
        elif 3*b < theta <= 5*b or -5*b < theta <= -3*b:
            return 'l'
        elif -3*b < theta <= -b or 5*b < theta <= 7*b:
            return 'd' 

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y)
                 ** 2 for node in node_list]
        minind = dlist.index(min(dlist)) # !!!!

        return minind

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    


def main():
    print("start " + __file__)

    # ====Search Path with RRT====
    Map = np.array([
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    rrt = RRT(Map=Map)
    sol = rrt.plan(start=[0, 0], goal=[0,19], animation=show_animation)
    path = sol['path']
    actions = sol['actions']
    print(path, actions)
    print(len(path), len(actions))


    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # Draw final path
        rrt.plot_plan(path = path)


if __name__ == '__main__':
    main()