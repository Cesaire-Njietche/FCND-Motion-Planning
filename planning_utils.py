from enum import Enum
from queue import PriorityQueue
import numpy as np
from sklearn.neighbors import KDTree
import networkx as nx
from shapely.geometry import Polygon, Point, LineString


def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1

    return grid, int(north_min), int(east_min)


# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)
    N_WEST = (-1, -1, np.sqrt(2))
    S_WEST = (1, -1, np.sqrt(2))
    N_EAST = (-1, 1, np.sqrt(2))
    S_EAST = (1, 1, np.sqrt(2))

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)
    if x -1 < 0 or y - 1 < 0 or grid[x - 1, y - 1] == 1:
        valid_actions.remove(Action.N_WEST)
    if x + 1 > n or y - 1 < 0 or grid[x + 1, y - 1] == 1:
        valid_actions.remove(Action.S_WEST)
    if x - 1 < 0 or y + 1 > m or grid[x - 1, y + 1] == 1:
        valid_actions.remove(Action.N_EAST)
    if x + 1 > n or y + 1 > m or grid[x + 1, y + 1] == 1:
        valid_actions.remove(Action.S_EAST) 

    return valid_actions


def a_star(grid, h, start, goal):

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False
    
    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]
            
        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))
             
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost



def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))
    #return np.sqrt((goal_position[0]-position[0])**2 + (goal_position[1]-position[1])**2)

def collinearity_int(p1, p2, p3): 
    collinear = False
    det = p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])
    if det == 0:
        collinear = True

    return collinear

def prune_path(path):
    """
    Prunes a given path by removing unnecessary waypoints. 
    """
    if path is not None:
        pruned_path = [p for p in path]
        i = 0
        j = 0
        while i < len(pruned_path) - 2:
            p1 = pruned_path[j]
            p2 = pruned_path[i+1]
            p3 = pruned_path[i+2]
            if(collinearity_int(p1,p2,p3)):
                del(pruned_path[i+1])
                i = i-1
            else:
                j = i+1
            i =i+1
    else:
        pruned_path = path
        
    return pruned_path

def can_connect_grid(p1, p2, polygons):
    """
    Given two points p1 and p2, checks whether a line between them does not cross a polygon
    """
    line = LineString([p1,p2])
    connect = True
    for (p, h) in polygons:
        if line.crosses(p) and 5 <= h:
            connect = False
            break
    return connect

def prune_path2(path, polygons):
    """
    Prunes the path by checking whether a line exist between two points 
    """
    i = 0
    j = 0
    while i < len(path) - 2:
        p1 = path[j]
        p2 = path[i+2]
        if(can_connect_grid(p1,p2,polygons)):
            del(path[i+1])
            i = i-1
        else:
            j = i+1
        i =i+1
    return path

def sample_points(data): 
    """
    Samples states in the environment and returns samples that are not in collision with obstacles
    """
    xmin = np.min(data[:, 0] - data[:, 3])
    xmax = np.max(data[:, 0] + data[:, 3])

    ymin = np.min(data[:, 1] - data[:, 4])
    ymax = np.max(data[:, 1] + data[:, 4])

    zmin = 0
    zmax = 20
    num_samples = 300

    xvals = np.random.uniform(xmin, xmax, num_samples)
    yvals = np.random.uniform(ymin, ymax, num_samples)
    zvals = np.random.uniform(zmin, zmax, num_samples)

    samples = np.array(list(zip(xvals, yvals, zvals)))

    tree = KDTree(data[:,0:3], leaf_size=2)
    free_samples = []
    for s in samples:
        inds = tree.query(s.reshape(1,-1), k=1, return_distance=False)[0]
   
        north, east, alt, d_north, d_east, d_alt = data[inds[0], :]
        obstacle = [north - d_north, north + d_north, east - d_east, east + d_east]
        corners = [(obstacle[0], obstacle[2]), (obstacle[0], obstacle[3]), (obstacle[1], obstacle[3]), (obstacle[1], obstacle[2])]
        
        height = alt + d_alt

        p = Polygon(corners)
        
        if p.contains(Point(s)):
            if  height < s[2]:
                free_samples.append(s) 
        else:
            free_samples.append(s)          
            
    return free_samples

def extract_polygons(data):
    """
    Extracts polygons from the obstacles
    """
    polygons = []
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        obstacle = [north - d_north, north + d_north, east - d_east, east + d_east]
        corners = [(obstacle[0], obstacle[2]), (obstacle[0], obstacle[3]), (obstacle[1], obstacle[3]), (obstacle[1], obstacle[2])]
        height = alt + d_alt

        p = Polygon(corners)
        polygons.append((p, height))

    return polygons

def can_connect(p1, p2, polygons):
    """
    Given two points p1 and p2, checks whether a line between them does not cross a polygon
    """
    line = LineString([p1,p2])
    connect = True
    for (p, h) in polygons:
        if line.crosses(p) and min(p1[2],p2[2]) <= h:
            connect = False
            break
    return connect

def create_graph(nodes, data):
    """
    Builds a graph from free samples with no polygons in between
    """
    G = nx.Graph()
    tree = KDTree(nodes, leaf_size=2)
    polygons = extract_polygons(data)
    for node in nodes:
        inds = tree.query(node.reshape(1,-1), k=10, return_distance=False)[0]
    
        for i in inds:
            n1 = tuple(node)
            n2 = tuple(nodes[i])
            if n1 == n2:
                continue
            if can_connect(n1,n2,polygons):
                G.add_edge(n1,n2,weight=np.linalg.norm(node - nodes[i]))
    return G

def closest_point(G, p):
    """
    Compute the closest point to the point p on the graph G
    """
    gp = np.array(G)
    n = np.linalg.norm(gp[:,0:2] - p, axis=1)
    ind = np.argmin(n)
    return gp[ind]


def a_star_graph(G, h, start, goal):
    """
    Builds a plan from start state to goal state on the graph
    """
    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False
    
    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]
            
        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for next_node in G[current_node]:
                branch_cost = current_cost + G.edges[current_node, next_node]['weight']
                queue_cost = branch_cost + h(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    branch[next_node] = (branch_cost, current_node, next_node)
                    queue.put((queue_cost, next_node))
             
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost