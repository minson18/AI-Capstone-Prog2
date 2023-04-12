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

# Return a set of tuple(x,y), which contains all blanks in mapStat
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
# Return a set of all possible steps
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
    def __init__(self, exploration_weight=5):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    # Coose the best successors of node
    def choose(self, node):
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

    # Train for one iteration
    def do_rollout(self, node):
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    # Find an unexplored successor of node
    def _select(self, node):
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    # Expand all children of node as a dict()
    def _expand(self, node):
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    # Return reward of node from a random simolation
    def _simulate(self, node):
        invert_reward = False
        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            invert_reward = not invert_reward

    # Send the reward back
    def _backpropagate(self, path, reward):
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    # Select a child of node using exploration weight
    def _uct_select(self, node):
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])
        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)

# Class of node of MCTS
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

# The node of this board game
class Board(Node):
    def __init__(self, last_step, turn, spaces, winner, terminal):
        """
        Parameters:
        last_step: the last step of this Board
        spaces: a set of all blanks on board
        turn: whether is the player turn or not
        winner: whether the winner is the player
        terminal: whether the state of this board is end game
        possible_steps: a seet of all possible steps of this board
        """
        self.last_step = last_step
        self.spaces = spaces
        self.turn = turn
        self.winner = winner
        self.terminal = terminal
        self.possible_steps = None
  
    # Find all children of this board
    def find_children(self):
        if self.terminal:  
            return set()
       
        if self.possible_steps is None:
            self.possible_steps = find_all_steps(self.spaces)
                
        return {self.make_move(step) for step in self.possible_steps}

    # FInd a random child of board
    def find_random_child(self):
        if self.terminal:
            return None 
        if self.possible_steps is None:
            self.possible_steps = find_all_steps(self.spaces)
            
        return self.make_move(choice(list(self.possible_steps)))

    # Return the reward when board is terminal
    # Player win, reward=1; player lose, reward=0
    def reward(self):
        if not self.terminal:
            raise RuntimeError(f"reward called on nonterminal board {self}")
            
        if self.winner != self.turn:
            raise RuntimeError(f"reward called on unreachable board {self}")
            
        if self.winner:
            return 1  
        else:
            return 0 
            
        raise RuntimeError(f"board has unknown winner type {self.winner}")

    # Return terminal state
    def is_terminal(self):
        return self.terminal      
 
    # Get the board after a step
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
    
    # Hash function, in case in MCTS, chidrens are store in dict()
    def __hash__(self):
        h = 0
        for i, j in self.spaces:
            h += 2**i + j
            
        if self.last_step is not None:
            h = h + self.last_step[0][0] * 2 + self.last_step[0][1] *3 + self.last_step[2]
        
        return int(h)
    
    # Compare function
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

# Miimax algorithm of alpha-beta pruning       
class AlphaBeta:
    def __init__(self, root):
        self.root = root  # Game node
        self.start = time.time() # start time of the search
        return

    # Start the search
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

    # Max agent
    def max_value(self, node, alpha, beta):     
        if self.isTerminal(node):
            return self.getReward(node)
            
        infinity = float('inf')
        value = -infinity
        
        # Avoid timeout error
        if (time.time() - self.start) > 4.5s:
            return infinity;           
        
        successors = self.getSuccessors(node)
        for state in successors:
            value = max(value, self.min_value(state, alpha, beta))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    # Min agent
    def min_value(self, node, alpha, beta):
        if self.isTerminal(node):
            return self.getReward(node)
            
        infinity = float('inf')
        value = infinity
        
        # Avoid timeout
        if (time.time() - self.start) > 4.5:
            return -infinity;         
        
        successors = self.getSuccessors(node)
        for state in successors:
            value = min(value, self.max_value(state, alpha, beta))
            if value <= alpha:
                return value
                
            beta = min(beta, value)

        return value

    # Get all chidrens of node
    def getSuccessors(self, node):
        assert node is not None
        return node.find_children()

    # Return terminal state
    def isTerminal(self, node):
        assert node is not None
        return node.is_terminal()

    # Get Reward
    def getReward(self, node):
        assert node is not None
        return node.reward()        
           
def Getstep(mapStat, gameStat):
    #Please write your code here
    #TODO
    # Initial the board of mapStat
    spaces = checkRemainMove(mapStat)
    board = Board(None, True, spaces, None, False)
    
    # When has more than 10 blanks, use MCTS
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
    
    # Otherwise use minimax
    else:
        tree = AlphaBeta(board)
        best_board = tree.alpha_beta_search(tree.root)
        
    return list(best_board.last_step)
  
# The testing function  
def play_game():    
    max_n = 5 # the size of board is n*n
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
