import STcpClient_1 as STcpClient
import numpy as np
import random

from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
import math
from random import choice
import copy
import time

'''
    input position (x,y) and direction
    output next node position on this direction
'''
def Next_Node(pos_x,pos_y,direction):
    if pos_y%2==1:
        if direction==1:
            return pos_x,pos_y-1
        elif direction==2:
            return pos_x+1,pos_y-1
        elif direction==3:
            return pos_x-1,pos_y
        elif direction==4:
            return pos_x+1,pos_y
        elif direction==5:
            return pos_x,pos_y+1
        elif direction==6:
            return pos_x+1,pos_y+1
    else:
        if direction==1:
            return pos_x-1,pos_y-1
        elif direction==2:
            return pos_x,pos_y-1
        elif direction==3:
            return pos_x-1,pos_y
        elif direction==4:
            return pos_x+1,pos_y
        elif direction==5:
            return pos_x-1,pos_y+1
        elif direction==6:
            return pos_x,pos_y+1

def checkRemainMove(mapStat):
    temp = set()
    rows = mapStat.shape[0]
    cols = mapStat.shape[1]
    for i in range(rows):
        for j in range(cols):
            if(mapStat[i][j] == 0):
                temp.add((i,j))
    return temp


'''
    輪到此程式移動棋子
    mapStat : 棋盤狀態(list of list), 為 12*12矩陣, 0=可移動區域, -1=障礙, 1~2為玩家1~2佔領區域
    gameStat : 棋盤歷史順序
    return Step
    Step : 3 elements, [(x,y), l, dir]
            x, y 表示要畫線起始座標
            l = 線條長度(1~3)
            dir = 方向(1~6),對應方向如下圖所示
              1  2
            3  x  4
              5  6
              
            0 0 0 
             0 0 0
            0 0 0
'''     

def find_all_steps(spaces):
    possible_steps = set()
    for space in spaces:
        possible_steps.add(((space[0], space[1]), 1, 1))
        x, y = space
        for direction in [1, 2, 3]: 
            i, j = space
            for l in [2, 3]:
                i, j = Next_Node(i, j, direction)
                if (i, j) in spaces:
                    possible_steps.add(((x, y), l, direction))  
                else:
                    break  
    return possible_steps
    
class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=5):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        invert_reward = False
        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            invert_reward = not invert_reward

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)

class Node():
    @abstractmethod
    def find_children(self):
        return set()

    @abstractmethod
    def find_random_child(self):
        return None

    @abstractmethod
    def is_terminal(self):
        return True

    @abstractmethod
    def reward(self):
        return 0

    @abstractmethod
    def __hash__(self):
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        return True

class Board(Node):
    def __init__(self, last_step, turn, spaces, winner, terminal):
        self.last_step = last_step
        self.spaces = spaces
        self.turn = turn
        self.winner = winner
        self.terminal = terminal
        self.possible_steps = None
        
    def find_children(self):
        if self.terminal:  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in each of the empty spots
        if self.possible_steps is None:
            self.possible_steps = find_all_steps(self.spaces)
                
        return {self.make_move(step) for step in self.possible_steps}

    def find_random_child(self):
        if self.terminal:
            return None  # If the game is finished then no moves can be made
        if self.possible_steps is None:
            self.possible_steps = find_all_steps(self.spaces)
            
        return self.make_move(choice(list(self.possible_steps)))

    def reward(self):
        if not self.terminal:
            raise RuntimeError(f"reward called on nonterminal board {self}")
        if self.winner != self.turn:
            # It's your turn and you've already won. Should be impossible.
            raise RuntimeError(f"reward called on unreachable board {self}")
        if self.winner:
            return 1  
        else:
            return 0 
        # The winner is neither True, False
        raise RuntimeError(f"board has unknown winner type {self.winner}")

    def is_terminal(self):
        return self.terminal      
    
    def make_move(self, step):
        spaces = copy.deepcopy(self.spaces)
        i, j = step[0]
        direction = step[2]
        for _ in range(step[1]):
            spaces.remove((i, j))
            i, j = Next_Node(i, j, direction)

        turn = not self.turn
        is_terminal = (len(spaces) == 0)
        if is_terminal: 
            winner = turn
        else:
            winner = None
            
        return Board(step, turn, spaces, winner, is_terminal)
    
    def __hash__(self):
        h = 0
        for i, j in self.spaces:
            h += 2**i + j
            
        if self.last_step is not None:
            h = h + self.last_step[0][0] * 2 + self.last_step[0][1] *3 + self.last_step[2]
        
        return int(h)
        
    def __eq__(node1, node2):
        if (node1.spaces != node2.spaces):
            return False
        if node1.turn  != node2.turn:
            return False
        if node1.last_step  != node2.last_step:
            return False
            
        return True
        
    def __ne__(node1, node2):
        return not __eq__(node1, node2)
        
class AlphaBeta:
    def __init__(self, root):
        self.root = root  # GameNode
        self.start = time.time()
        return

    def alpha_beta_search(self, node):
        infinity = float('inf')
        best_val = -infinity
        beta = infinity

        successors = self.getSuccessors(node)
        best_state = None
        for state in successors:
            value = self.min_value(state, best_val, beta)
            if value > best_val:
                best_val = value
                best_state = state
        return best_state

    def max_value(self, node, alpha, beta):     
        if self.isTerminal(node):
            return self.getUtility(node)
            
        infinity = float('inf')
        value = -infinity
        if (time.time() - self.start) > 4.5:
            return infinity;           
        
        successors = self.getSuccessors(node)
        for state in successors:
            value = max(value, self.min_value(state, alpha, beta))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def min_value(self, node, alpha, beta):
        if self.isTerminal(node):
            return self.getUtility(node)
            
        infinity = float('inf')
        value = infinity
        if (time.time() - self.start) > 4.5:
            return -infinity;         
        
        successors = self.getSuccessors(node)
        for state in successors:
            value = min(value, self.max_value(state, alpha, beta))
            if value <= alpha:
                return value
            beta = min(beta, value)

        return value
        
    def getSuccessors(self, node):
        assert node is not None
        return node.find_children()

    def isTerminal(self, node):
        assert node is not None
        return node.is_terminal()

    def getUtility(self, node):
        assert node is not None
        return node.reward()        
           
def Getstep(mapStat, gameStat):
    #Please write your code here
    #TODO
    spaces = checkRemainMove(mapStat)
    board = Board(None, True, spaces, None, False)
    
    if len(spaces) > 10:
        start = time.time()
        tree = MCTS()
        iteration = 0
        while True:
            tree.do_rollout(board)
            iteration += 1
            if (time.time() - start) > 4:
                break
        print(f"iteration = {iteration}")
        best_board = tree.choose(board)
    
    else:
        tree = AlphaBeta(board)
        best_board = tree.alpha_beta_search(tree.root)
        
    return list(best_board.last_step)
    
def play_game():    
    max_n = 5
    mapStat = np.zeros((max_n, max_n))
    for row in mapStat:
        print(row)
    print('start game')
    player = 0
    while True:
        player += 1   
        step = Getstep(mapStat, None)
        i, j = step[0]
        direction = step[2]
        for t in range(step[1]):
            mapStat[i][j] = player
            i, j = Next_Node(i, j, direction)
            
        print(f"Player: {player}, Step: {step}")
        n = np.array(mapStat)
        for y, row in enumerate(n.T):
            if y % 2 == 0:
                print(f"{row}")
            else:
                print(f" {row}")
        print("")    
        
        spaces = checkRemainMove(mapStat)
        is_terminal = (len(spaces) == 0)
        if is_terminal:
            break
        player = (player) % 2
    
#play_game()


# start game
print('start game')
while (True):

    (end_program, id_package, mapStat, gameStat) = STcpClient.GetBoard()
    if end_program:
        STcpClient._StopConnect()
        break
    
    decision_step = Getstep(mapStat, gameStat)
    
    STcpClient.SendStep(id_package, decision_step)
