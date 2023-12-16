import sys
import time
import threading

import numpy as np
import torch

import utils

import logging
from datetime import datetime
logging.basicConfig(
    filename='logs/log_'+datetime.now().strftime('%y%m%d')+'.txt',
    level=logging.WARNING)

PRINT_MCTS = False
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class Agent(object):
    def __init__(self, board_size):

        self.policy = np.zeros(board_size**2, 'float')
        self.visit = np.zeros(board_size**2, 'float')
        self.message = 'Hello'

    def get_policy(self):
        return self.policy

    def get_visit(self):
        return self.visit

    def get_name(self):
        return type(self).__name__

    def get_message(self):
        return self.message

    def get_pv(self, root_id):
        return None, None

class MinmaxAgent(Agent):
    def __init__(self, board_size):
        super().__init__(board_size)
        self.board_size = board_size
        self.root_id = None
        self.win_mark=5
        self.initialize_game()
        self.initial_check = 0
        self.stack =0
        self.search_depth = 5

    def initialize_game(self):
        self.current_state = np.array([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                              [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                              [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                              [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                              [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                              [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                              [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                              [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                              [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]) 
        
    def max(self,root_id, alpha, beta,search_depth):    
        if self.initial_check == 0:
            for i,id in enumerate(root_id):
                row, col = divmod(id, 9)
                if i==1:
                    self.current_state[0][0]=0
                if i%2==0:
                    self.current_state[row][col] = -1.0
                else:
                    self.current_state[row][col] = 1.0
                self.initial_check+=1  
        
        maxv = -2
        px = None
        py = None
        result = utils.check_win1(self.current_state, self.win_mark)
        if search_depth <= 0:  
            search_depth = self.search_depth
            if len(root_id) % 2 == 0: 
                if result == 1:
                    self.stack+=1    
                    if self.stack % 10000==0: 
                        logging.warning((self.stack))        
                    return (-1, 0, 0)
                elif result == 2:
                    self.stack+=1
                    if self.stack % 10000==0: 
                        logging.warning((self.stack))                    
                    return (1, 0, 0)
                elif result == 3:
                    self.stack+=1
                    if self.stack % 10000==0: 
                        logging.warning((self.stack))                    
                    return (0, 0, 0)
                else:
                    self.stack+=1
                    if self.stack % 10000==0: 
                        logging.warning((self.stack))
                    
                    return (0,0,0)
            else: 
                if result == 1:
                    self.stack+=1        
                    if self.stack % 10000==0: 
                        logging.warning((self.stack))           
                    return (1, 0, 0)
                elif result == 2:
                    self.stack+=1     
                    if self.stack % 10000==0: 
                        logging.warning((self.stack))                
                    return (-1, 0, 0)
                elif result == 3:
                    self.stack+=1
                    if self.stack % 10000==0: 
                        logging.warning((self.stack))                     
                    return (0, 0, 0)
                else:
                    self.stack+=1
                    if self.stack % 10000==0: 
                        logging.warning((self.stack)) 
                    return (0,0,0)

        if len(root_id) % 2 == 0: 
            if result == 1:
                self.stack+=1    
                if self.stack % 10000==0: 
                    logging.warning((self.stack))        
                return (-1, 0, 0)
            elif result == 2:
                self.stack+=1
                if self.stack % 10000==0: 
                    logging.warning((self.stack))                    
                return (1, 0, 0)
            elif result == 3:
                self.stack+=1
                if self.stack % 10000==0: 
                    logging.warning((self.stack))                    
                return (0, 0, 0)
            
        else: 
            if result == 1:
                self.stack+=1        
                if self.stack % 10000==0: 
                    logging.warning((self.stack))           
                return (1, 0, 0)
            elif result == 2:
                self.stack+=1     
                if self.stack % 10000==0: 
                    logging.warning((self.stack))                
                return (-1, 0, 0)
            elif result == 3:
                self.stack+=1
                if self.stack % 10000==0: 
                    logging.warning((self.stack))                     
                return (0, 0, 0)

        for i in range(0, self.board_size):
            for j in range(0, self.board_size):
                if self.current_state[i][j] == 0.0:
                    if len(root_id) %2 ==0:
                        self.current_state[i][j] = -1.0
                    else:
                        self.current_state[i][j] = 1.0
                    search_depth -=1
                    (m, qx, qy) = self.min(root_id,-2,2,search_depth)
                    if m > maxv:
                        maxv = m
                        px = i
                        py = j
                    self.current_state[i][j] = 0.0

                    if maxv >= beta:
                        return (maxv, px, py)

                    if maxv > alpha:
                        alpha = maxv

        return (maxv, px, py)
            
    def min(self, root_id, alpha, beta,search_depth):
        minv = 2
        qx = None
        qy = None
        result = utils.check_win1(self.current_state, self.win_mark)
        if search_depth <= 0:  
            search_depth = self.search_depth
            if len(root_id) % 2 == 0: 
                if result == 1:
                    self.stack+=1    
                    if self.stack % 10000==0: 
                        logging.warning((self.stack))        
                    return (-1, 0, 0)
                elif result == 2:
                    self.stack+=1
                    if self.stack % 10000==0: 
                        logging.warning((self.stack))                    
                    return (1, 0, 0)
                elif result == 3:
                    self.stack+=1
                    if self.stack % 10000==0: 
                        logging.warning((self.stack))                    
                    return (0, 0, 0)
                else:
                    self.stack+=1
                    if self.stack % 10000==0: 
                        logging.warning((self.stack))
                    
                    return (0,0,0)
            else: 
                if result == 1:
                    self.stack+=1        
                    if self.stack % 10000==0: 
                        logging.warning((self.stack))           
                    return (1, 0, 0)
                elif result == 2:
                    self.stack+=1     
                    if self.stack % 10000==0: 
                        logging.warning((self.stack))                
                    return (-1, 0, 0)
                elif result == 3:
                    self.stack+=1
                    if self.stack % 10000==0: 
                        logging.warning((self.stack))                     
                    return (0, 0, 0)
                else:
                    self.stack+=1
                    if self.stack % 10000==0: 
                        logging.warning((self.stack)) 
                    return (0,0,0)

        if len(root_id) % 2 == 0: 
            if result == 1:
                self.stack+=1    
                if self.stack % 10000==0: 
                    logging.warning((self.stack))        
                return (-1, 0, 0)
            elif result == 2:
                self.stack+=1
                if self.stack % 10000==0: 
                    logging.warning((self.stack))                    
                return (1, 0, 0)
            elif result == 3:
                self.stack+=1
                if self.stack % 10000==0: 
                    logging.warning((self.stack))                    
                return (0, 0, 0)
            
        else: 
            if result == 1:
                self.stack+=1        
                if self.stack % 10000==0: 
                    logging.warning((self.stack))           
                return (1, 0, 0)
            elif result == 2:
                self.stack+=1     
                if self.stack % 10000==0: 
                    logging.warning((self.stack))                
                return (-1, 0, 0)
            elif result == 3:
                self.stack+=1
                if self.stack % 10000==0: 
                    logging.warning((self.stack))                     
                return (0, 0, 0)
        for i in range(0, self.board_size):
            for j in range(0, self.board_size):
                if self.current_state[i][j] == 0.0:
                    if len(root_id) %2 ==0:
                        self.current_state[i][j] = 1.0
                    else:
                        self.current_state[i][j] = -1.0
                    search_depth -= 1
                    (m, qx, qy) = self.max(root_id,-2,2,search_depth) # -1 0 0
                    if m < minv:
                        minv = m
                        qx = i
                        qy = j
                    self.current_state[i][j] = 0.0

                    if minv <= alpha:
                        return (minv, qx, qy)

                    if minv < beta:
                        beta = minv

        return (minv, qx, qy)

    def get_random_pi(self,root_id):
        self.root_id = root_id
        pi = np.full(81, 1/81)
        return pi

    def get_pi(self, root_id):
        self.root_id = root_id
        (_, px, py) = self.max(root_id,-2,2,self.search_depth)
        self.stack = 0
        self.initial_check=0
        self.initialize_game()
        action_index = px*9+py
        matrix = [[0] * 9 for _ in range(9)]
        matrix[px][py] = 1
        pi = matrix
        return pi, action_index

    def reset(self):
        self.root_id = None

    def del_parents(self, root_id):
        return 

class ZeroAgent(Agent):
    def __init__(self, board_size, num_mcts, inplanes, noise=True):
        super(ZeroAgent, self).__init__(board_size)
        self.board_size = board_size
        self.num_mcts = num_mcts
        self.inplanes = inplanes
        # tictactoe and omok
        self.win_mark = 3 if board_size == 3 else 5
        self.alpha = 10 / self.board_size**2
        self.c_puct = 5
        self.noise = noise
        self.root_id = None
        self.model = None
        self.tree = {}
        self.is_real_root = True

    def reset(self):
        self.root_id = None
        self.tree.clear()
        self.is_real_root = True

    def get_pi(self, root_id, tau):
        self._init_mcts(root_id)
        self._mcts(self.root_id)

        visit = np.zeros(self.board_size**2, 'float')
        policy = np.zeros(self.board_size**2, 'float')

        for action_index in self.tree[self.root_id]['child']:
            child_id = self.root_id + (action_index,)
            visit[action_index] = self.tree[child_id]['n']
            policy[action_index] = self.tree[child_id]['p']

        self.visit = visit
        self.policy = policy

        pi = visit / visit.sum()

        if tau == 0:
            pi, _ = utils.argmax_onehot(pi)
        return pi
    
    def get_random_pi(self,root_id):
        self._init_mcts(root_id)
        #self._mcts(self.root_id)

        pi = np.full(81, 1/81)
        return pi
    
    def _init_mcts(self, root_id):
        self.root_id = root_id
        if self.root_id not in self.tree:
            self.is_real_root = True
            # init root node
            self.tree[self.root_id] = {'child': [],
                                       'n': 0.,
                                       'w': 0.,
                                       'q': 0.,
                                       'p': 0.}
        # add noise
        else:
            self.is_real_root = False
            if self.noise:
                children = self.tree[self.root_id]['child']
                noise_probs = np.random.dirichlet(
                    self.alpha * np.ones(len(children)))

                for i, action_index in enumerate(children):
                    child_id = self.root_id + (action_index,)
                    self.tree[child_id]['p'] = 0.75 * \
                        self.tree[child_id]['p'] + 0.25 * noise_probs[i]

    def _mcts(self, root_id):
        start = time.time()
        if self.is_real_root:
            # do not count first expansion of the root node
            num_mcts = self.num_mcts + 1
        else:
            num_mcts = self.num_mcts

        for i in range(num_mcts):

            if PRINT_MCTS:
                sys.stdout.write('simulation: {}\r'.format(i + 1))
                sys.stdout.flush()

            self.message = 'simulation: {}\r'.format(i + 1)

            # selection
            leaf_id, win_index = self._selection(root_id)

            # expansion and evaluation
            value, reward = self._expansion_evaluation(leaf_id, win_index)

            # backup
            self._backup(leaf_id, value, reward)

        finish = time.time() - start
        if PRINT_MCTS:
            print("{} simulations end ({:0.0f}s)".format(i + 1, finish))

    def _selection(self, root_id):
        node_id = root_id

        while self.tree[node_id]['n'] > 0:
            board = utils.get_board(node_id, self.board_size)
            win_index = utils.check_win(board, self.win_mark)

            if win_index != 0:
                return node_id, win_index

            qu = {}
            ids = []
            total_n = 0

            for action_idx in self.tree[node_id]['child']:
                edge_id = node_id + (action_idx,)
                n = self.tree[edge_id]['n']
                total_n += n

            for i, action_index in enumerate(self.tree[node_id]['child']):
                child_id = node_id + (action_index,)
                n = self.tree[child_id]['n']
                q = self.tree[child_id]['q']
                p = self.tree[child_id]['p']
                u = self.c_puct * p * np.sqrt(total_n) / (n + 1)
                qu[child_id] = q + u

            max_value = max(qu.values())
            ids = [key for key, value in qu.items() if value == max_value]
            node_id = ids[np.random.choice(len(ids))]

        board = utils.get_board(node_id, self.board_size)
        win_index = utils.check_win(board, self.win_mark)

        return node_id, win_index

    def _expansion_evaluation(self, leaf_id, win_index):
        leaf_state = utils.get_state_pt(
            leaf_id, self.board_size, self.inplanes)
        self.model.eval()
        with torch.no_grad():
            state_input = torch.tensor(np.array([leaf_state])).to(device).float()
            policy, value = self.model(state_input)
            policy = policy.cpu().numpy()[0]
            value = value.cpu().numpy()[0]

        if win_index == 0:
            # expansion
            actions = utils.legal_actions(leaf_id, self.board_size)
            prior_prob = np.zeros(self.board_size**2)

            # re-nomalization
            for action_index in actions:
                prior_prob[action_index] = policy[action_index]

            prior_prob /= prior_prob.sum()

            if self.noise:
                # root node noise
                if leaf_id == self.root_id:
                    noise_probs = np.random.dirichlet(
                        self.alpha * np.ones(len(actions)))                    

            for i, action_index in enumerate(actions):
                child_id = leaf_id + (action_index,)

                prior_p = prior_prob[action_index]

                if self.noise:
                    if leaf_id == self.root_id:
                        prior_p = 0.75 * prior_p + 0.25 * noise_probs[i]

                self.tree[child_id] = {'child': [],
                                       'n': 0.,
                                       'w': 0.,
                                       'q': 0.,
                                       'p': prior_p}
                
                self.tree[leaf_id]['child'].append(action_index)
            # return value
            reward = False
            return value, reward
        else:
            # terminal node
            # return reward
            reward = 1.
            value = False
            return value, reward

    def _backup(self, leaf_id, value, reward):
        node_id = leaf_id
        count = 0
        while node_id != self.root_id[:-1]:
            self.tree[node_id]['n'] += 1

            if not reward:
                self.tree[node_id]['w'] += (-value) * (-1)**(count)
                count += 1
            else:
                self.tree[node_id]['w'] += reward * (-1)**(count)
                count += 1

            self.tree[node_id]['q'] = (self.tree[node_id]['w'] /
                                       self.tree[node_id]['n'])
            parent_id = node_id[:-1]
            node_id = parent_id

    def del_parents(self, root_id):
        max_len = 0
        if self.tree:
            for key in list(self.tree.keys()):
                if len(key) > max_len:
                    max_len = len(key)
                if len(key) < len(root_id):
                    del self.tree[key]
        print('tree size:', len(self.tree))
        print('tree depth:', 0 if max_len <= 0 else max_len - 1)

    def get_pv(self, root_id):
        state = utils.get_state_pt(root_id, self.board_size, self.inplanes)
        self.model.eval()
        with torch.no_grad():
            state_input = torch.tensor([state]).to(device).float()
            policy, value = self.model(state_input)
            p = policy.data.cpu().numpy()[0]
            v = value.data.cpu().numpy()[0]
        return p, v

class HumanAgent(Agent):
    COLUMN = {"a": 0, "b": 1, "c": 2,
              "d": 3, "e": 4, "f": 5,
              "g": 6, "h": 7, "i": 8,
              "j": 9, "k": 10, "l": 11,
              "m": 12, "n": 13, "o": 14}

    def __init__(self, board_size, env):
        super(HumanAgent, self).__init__(board_size)
        self.board_size = board_size
        self._init_board_label()
        self.root_id = (0,)
        self.env = env

    def get_pi(self, root_id, board, turn, tau):
        self.root_id = root_id

        while True:
            action = 0

            _, check_valid_pos, _, _, action_index = self.env.step(
                action)

            if check_valid_pos is True:
                pi = np.zeros(self.board_size**2, 'float')
                pi[action_index] = 1
                break

        return pi

    def _init_board_label(self):
        self.last_label = str(self.board_size)

        for k, v in self.COLUMN.items():
            if v == self.board_size - 1:
                self.last_label += k
                break

    def input_action(self, last_label):
        action_coord = input('1a ~ {}: '.format(last_label)).rstrip().lower()
        row = int(action_coord[0]) - 1
        col = self.COLUMN[action_coord[1]]
        action_index = row * self.board_size + col
        return action_index

    def reset(self):
        self.root_id = None

    def del_parents(self, root_id):
        return


