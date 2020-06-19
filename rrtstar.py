"""

Path planning Sample Code with RRT*

author: Atsushi Sakai(@Atsushi_twi)

"""

import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from rrt import RRT

show_animation = False

class RRTStar(RRT):
    """
    Class for RRT Star planning
    """

    class Node(RRT.Node):
        def __init__(self, x, y):
            super().__init__(x, y)
            self.cost = 0.0

        def eq(self, node):
            if self.x == node.x and self.y == node.y and self.parent is not None and node.parent is not None and self.parent.x == node.parent.x and self.parent.y == node.parent.y:
                return True
            return False

        def coord(self):
            print("Position <%d,%d>, cost: %d."%(self.x,self.y, self.cost))

    def __init__(self, start, goal, Map,
                 expand_dis=1.0,
                 path_resolution=1.0,
                 goal_sample_rate=20,
                 max_iter=10000,
                 connect_circle_dist=1.0,
                 verbose = False
                 ):
        super().__init__(start, goal, Map, expand_dis, path_resolution, goal_sample_rate, max_iter)
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        """
        self.connect_circle_dist = connect_circle_dist
        self.goal_node = self.Node(goal[0], goal[1])
        self.best_cost = np.Inf
        self.list_size = 0
        self.count_convrg = 0
        self.verbose = verbose


    def planning(self, animation=True, search_until_max_iter=True):
        """
        rrt star path planning

        animation: flag for animation on or off
        search_until_max_iter: search until max iteration for path improving or not
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            
            if i % 100:
                i_best_cost = self.search_best_goal_node() 
                best_cost_so_far = self.node_list[i_best_cost].cost if i_best_cost is not None else np.Inf

                if best_cost_so_far < self.best_cost or len(self.node_list) > self.list_size:
                    self.best_cost = best_cost_so_far
                    self.count_convrg = 0
                    self.list_size = len(self.node_list)
                elif best_cost_so_far is not np.Inf:
                    self.count_convrg += 1

            if self.verbose:
                print("Iter:", i, ", number of nodes:", len(self.node_list), ", best cost: ", self.best_cost+2)

            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd, self.expand_dis)

            if self.check_collision(new_node):
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)
                if new_node and not self.check_exist(new_node):
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_inds, rewire_near = False)

            if animation and i % 100 == 0:
                self.draw_graph(rnd)

            if ((not search_until_max_iter) and new_node) or self.count_convrg > 500:  # check reaching the goal
                last_index = self.search_best_goal_node()
                if last_index:
                    return self.generate_final_course(last_index)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            return self.generate_final_course(last_index)

        return None

    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return None

        # search nearest cost in near_inds
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(t_node):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.parent = self.node_list[min_ind]
        new_node.cost = min_cost

        return new_node

    def check_exist(self, new_node):
        for n in self.node_list:
            if new_node.eq(n) and new_node.cost == n.cost:
                return True

        return False

    def search_best_goal_node(self):
        dist_to_goal_list = [self.calc_dist_to_goal(n.x, n.y) for n in self.node_list]
        goal_inds = [i for i, x in enumerate(dist_to_goal_list) if x == 0]

        if not goal_inds:
            return None

        min_cost = min([self.node_list[i].cost for i in goal_inds])
        for i in goal_inds:
            if self.node_list[i].cost == min_cost:
                return i

        return None

    def find_near_nodes(self, new_node):
        r = self.connect_circle_dist
        dist_list = [np.abs(node.x - new_node.x) + np.abs(node.y - new_node.y) for node in self.node_list]
        near_inds = [i for i, x in enumerate(dist_list) if x == r ]
        return near_inds

    def rewire(self, new_node, near_inds, rewire_near = False):
        for i in near_inds:
            near_node = self.node_list[i]
            new_cost = self.calc_new_cost(new_node, near_node)
            improved_cost = new_cost < near_node.cost 

            if improved_cost:
                self.node_list[i].parent = new_node
                self.node_list[i].cost = new_cost
                self.propagate_cost_to_leaves(new_node)

    def rewire_all(self):
        last_index = self.search_best_goal_node()
        if not last_index:
            return
        best_cost_so_far = self.node_list[last_index].cost 
        print("Rewiring all...")

        for i, node in enumerate(self.node_list):
            if np.random.rand() < 1:
                near_inds = self.find_near_nodes(node)
                try:
                    j = near_inds.index(i)
                    del near_inds[j]
                except:
                    pass
                self.rewire(node, near_inds)

        last_index = self.search_best_goal_node()
        best_cost = self.node_list[last_index].cost 
        print("Rewired all with cost from %d to %d."%(best_cost_so_far, best_cost))

    def calc_new_cost(self, from_node, to_node):
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):

        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)


def main():
    print("Start " + __file__)

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

    # Set Initial parameters
    rrt_star = RRTStar(start=[0, 0],
                    goal=[0,19],
                    Map=Map)
    sol = rrt_star.planning(animation=show_animation)
    path = sol['path']
    actions = sol['actions']
    print(path, actions)
    print(len(path), len(actions))

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")
        print("Length: ", len(path))

        # Draw final path
        rrt_star.draw_graph(path = path)


if __name__ == '__main__':
    main()