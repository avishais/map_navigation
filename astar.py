import numpy as np
import matplotlib.pyplot as plt

# https://www.analytics-link.com/post/2018/09/14/applying-the-a-path-finding-algorithm-in-python-part-1-2d-square-grid

class Node():
    # node class for A* Pathfinding

    def __init__(self, parent=None, position=None, parent_action=None):
        self.parent = parent
        self.position = position
        self.parent_action = parent_action

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return np.all(self.position == other.position)

class Astar():
    # directions = [[0,1],[0,-1],[1,0],[-1,0]]#,[1,1],[1,-1],[-1,1],[-1,-1]]
    directions = {'u': np.array([-1,0]), 'd': np.array([1,0]), 'r': np.array([0,1]), 'l': np.array([0,-1])}

    def __init__(self, Map = []):
        self.Map = Map
        self.set_g_func()

    def set_map(self, Map):
        self.Map = Map

    # def set_score_map(self, score_map):
    #     self.score_map = score_map
        
    # heuristic function for path scoring
    def heuristic(self, a, b):
        return (b[0] - a[0]) + (b[1] - a[1])
        # return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    def cost(self, pos):
        return 1

    def path_length(self, path):
        L = 0
        for i in range(1, len(path)):
            L += self.heuristic(path[i-1], path[i])

        return L

    def set_g_func(self, g_func = None):
        if g_func is None:
            self.g_func = self.cost
        else:
            self.g_func = g_func
 
    # path finding function
    def plan(self, start, goal):

        close_list = []

        end_node = Node(None, goal, None)

        start_node = Node(None, start, None)
        start_node.g = 0
        start_node.h = self.heuristic(start_node.position, end_node.position)
        
        open_list = []

        open_list.append(start_node)

        while open_list:

            current_node = open_list[0]
            current_index = 0
            for index, item in enumerate(open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index
            open_list.pop(current_index)

            if np.all(current_node == end_node):
                path = []
                action_path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    action_path.append(current.parent_action)
                    current = current.parent
                print('Final cost: ', current_node.g, ", solution length: ", len(path))
                return {'path': path[::-1], 'actions': action_path[::-1][1:]}

            close_list.append(current_node)

            for d in self.directions.keys():
                neighbor = [current_node.position[0] + self.directions[d][0], current_node.position[1] + self.directions[d][1]]

                if not (0 <= neighbor[0] < self.Map.shape[0] and 0 <= neighbor[1] < self.Map.shape[1] and self.Map[neighbor[0]][neighbor[1]] == 0):
                    continue

                neighbor_node = Node(current_node, neighbor, d)
                
                g = self.g_func(neighbor_node.position)
                neighbor_node.g = current_node.g + g 

                # Neigbor is on the closed list
                flag = False
                for closed_node in close_list:
                    if closed_node == neighbor_node and neighbor_node.g >= closed_node.g: # Not sure I should check this
                        flag = True
                        break
                if flag:
                    del neighbor_node
                    continue

                # Child is already in the open list
                flag = False
                for open_node in open_list:
                    if neighbor_node == open_node and neighbor_node.g >= open_node.g:
                        flag = True
                        break
                if flag:
                    del neighbor_node
                    continue

                # Add the child to the open list
                neighbor_node.f = neighbor_node.g + self.heuristic(current_node.position, neighbor_node.position)
                # print("Nei: ", neighbor_node.f, g, neighbor_node.g, self.heuristic(current_node.position, neighbor_node.position))
                open_list.append(neighbor_node)

        return False

    def plot_plan(self, route, do_plot = True):
        route = np.array(route)
        start = route[0]
        goal = route[-1]

        # plot map and path
        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(self.Map, cmap=plt.cm.Dark2)
        ax.scatter(start[1],start[0], marker = "*", color = "yellow", s = 200)
        ax.scatter(goal[1],goal[0], marker = "*", color = "red", s = 200)
        ax.plot(route[:,1], route[:,0], color = "black")

        if do_plot:
            plt.show()





##############################################################################

def alt_cost(pos):
    return (pos[0]-17)**2/10

def main():
    print("Start " + __file__)

    grid = np.array([
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


    # start point and goal
    start = [0,0]
    goal = [0,19]

    E = Astar(Map = grid)
    # E.set_g_func(alt_cost)
    print(start)

    sol = E.plan(start, goal)
    route = sol['path']
    actions = sol['actions']

    print(len(route), len(actions))
    if route:
        print(E.path_length(route))
        E.plot_plan(route)
    else:
        print("Path not found!")

if __name__ == '__main__':
    main()


 

