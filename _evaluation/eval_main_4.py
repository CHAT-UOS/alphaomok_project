import os 
import sys
import numpy as np
import torch

import agents

import ranmini1_model
import model_seran_tuned
import model

import utils

# env_small: 9x9, env_regular: 15x15
from env import env_small as game

# Web API
import logging
from datetime import datetime
logging.basicConfig(
    filename='logs/log_'+datetime.now().strftime('%y%m%d')+'.txt',
    level=logging.WARNING)

from info.agent_info import AgentInfo
from info.game_info import GameInfo

PRINT_SELFPLAY = False
USE_TENSORBOARD = True
if USE_TENSORBOARD:
    from tensorboardX import SummaryWriter
    Writer1 = SummaryWriter("elo/minmax5_ranmini400/ranmini400")
    Writer2 = SummaryWriter("elo/minmax5_ranmini400/minmax5")

BOARD_SIZE = game.Return_BoardParams()[0]

N_BLOCKS_PLAYER1 = 5
N_BLOCKS_PLAYER2 = 5


IN_PLANES_PLAYER1 = 5  
IN_PLANES_PLAYER2 = 5


OUT_PLANES_PLAYER1 = 128
OUT_PLANES_PLAYER2 = 128


N_MCTS_PLAYER1 = 400
N_MCTS_PLAYER2 = 100

N_MATCH = 50

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

class Evaluator(object):
    def __init__(self):
        self.player1 = None
        self.player2 = None
        pass

    def set_agents(self, black,black_id,white_id):
        game_mode = 'text'

        self.env = game.GameState(game_mode)

        self.player1 = agents.ZeroAgent(BOARD_SIZE,
                                        N_MCTS_PLAYER1,
                                        IN_PLANES_PLAYER1,
                                        noise=False)
        
        if black_id == 'seran_tuned':
            self.player1.model = model_seran_tuned.PVNet(N_BLOCKS_PLAYER1,
                                            IN_PLANES_PLAYER1,
                                            OUT_PLANES_PLAYER1,
                                            BOARD_SIZE).to(device)
            
        if black_id == 'ranmini1':
            self.player1.model = ranmini1_model.PVNet(N_BLOCKS_PLAYER1,
                                                IN_PLANES_PLAYER1,
                                                OUT_PLANES_PLAYER1,
                                                BOARD_SIZE).to(device)

        else:
            self.player1.model = model.PVNet(N_BLOCKS_PLAYER1,
                                            IN_PLANES_PLAYER1,
                                            OUT_PLANES_PLAYER1,
                                            BOARD_SIZE).to(device)
        
        state_a = self.player1.model.state_dict()
        my_state_a = torch.load(
            black, map_location='cuda:0' if use_cuda else 'cpu')
        for k, v in my_state_a.items():
            if k in state_a:
                state_a[k] = v
        self.player1.model.load_state_dict(state_a)

        

        self.player2 = agents.MinmaxAgent(BOARD_SIZE)

    def get_action(self, root_id, board, turn, enemy_turn,time_step):
        if turn != enemy_turn:
            if time_step <= 2:
                pi = self.player1.get_random_pi(root_id)
            else:
                pi = self.player1.get_pi(root_id, tau=0)
            
            action, action_index = utils.argmax_onehot(pi)
        else:
            if time_step <= 2:
                pi = self.player2.get_random_pi(root_id)
                action, action_index = utils.argmax_onehot(pi)
            else:
                action,action_index  = self.player2.get_pi(root_id)
        return action, action_index

    def return_env(self):
        return self.env

    def reset(self):
        self.player1.reset()
        self.player2.reset()

    def put_action(self, action_idx, turn, enemy_turn):
        
        if PRINT_SELFPLAY == True:
            print(self.player1)
            logging.warning((self.player1))

        if turn != enemy_turn:
            if type(self.player1) is agents.WebAgent:
                self.player1.put_action(action_idx)
        else:
            if type(self.player2) is agents.WebAgent:
                self.player2.put_action(action_idx)


def elo(player1_elo, player2_elo, p_winscore, e_winscore):
    elo_diff = player2_elo - player1_elo
    ex_pw = 1 / (1 + 10**(elo_diff / 400))
    ex_ew = 1 / (1 + 10**(-elo_diff / 400))
    player1_elo += 32 * (p_winscore - ex_pw)
    player2_elo += 32 * (e_winscore - ex_ew)

    return player1_elo, player2_elo


evaluator = Evaluator()

def main(player1_model_id,player2_model_id,player1_model_path,player1_elo,Minmax_elo,step):
    turn = 0
    enemy_turn = 1

    elo1 = "player1_elo"
    elo2 = "player2_elo"

    black = player1_model_path

    black_id = player1_model_id
    white_id = player2_model_id

    black_elo = elo1
    white_elo = elo2

    Player1_win = 0
    Player2_win = 0
    draw = 0
    
    evaluator.set_agents(black,black_id,white_id)
    env = evaluator.return_env()
    
    for i in range(N_MATCH):
        step+=1
        result = {black_id: 0, white_id: 0, 'Draw': 0}
        board = np.zeros([BOARD_SIZE, BOARD_SIZE])
        root_id = (0,)
        win_index = 0
        action_index = None

        if PRINT_SELFPLAY == True:
            if i % 2 == 0:
                print(f'{black_id} Color: Black')
            else:
                print(f'{black_id} Color: White')

        # 0:Running 1:Player1 Win, 2: player2 Win 3: Draw
        time_step = 0
        while win_index == 0:
            time_step += 1
            
            utils.render_str(board, BOARD_SIZE, action_index)

            action, action_index = evaluator.get_action(root_id,
                                                        board,
                                                        turn,
                                                        enemy_turn,
                                                        time_step)
            if turn != enemy_turn:
                # player1 turn
                root_id = evaluator.player1.root_id + (action_index,)
            else:
                # player2 turn
                root_id = evaluator.player2.root_id + (action_index,)

            board, check_valid_pos, win_index, turn, _ = env.step(action)

            move = np.count_nonzero(board)
            if turn == enemy_turn:
                evaluator.player2.del_parents(root_id)

            else:
                evaluator.player1.del_parents(root_id)

            if win_index != 0:
                # 0:Running 1:Player1 Win, 2: player2 Win 3: Draw
                if turn == enemy_turn:
                    if win_index == 3:
                        print('\nDraw!')
                        draw +=1
                        player1_elo, _ = elo(
                            player1_elo, Minmax_elo, 0.5, 0.5)
                        
                    else:
                        print(f'\n{black_id} Win!')
                        Player1_win += 1
                        player1_elo, _ = elo(
                            player1_elo, Minmax_elo, 1, 0)

                else:
                    if win_index == 3:
                        print('\nDraw!')
                        draw +=1
                        player1_elo, _ = elo(
                            player1_elo, Minmax_elo, 0.5, 0.5)
                            
                    else:
                        print(f'\n{white_id} Win!')
                        Player2_win += 1
                        player1_elo, _ = elo(
                            player1_elo, Minmax_elo, 0, 1)
                            
                utils.render_str(board, BOARD_SIZE, action_index)

                enemy_turn = abs(enemy_turn - 1)
                turn = 0

                if PRINT_SELFPLAY == True:

                    print(('=' * 20, " {}  Game End  ".format(i + 1), '=' * 20))
                    print((f"Player_win: {Player1_win}, enemy_win: {Player2_win}, Draw: {draw}"))
                    print((f"player1_elo: {player1_elo},Minmax_elo: {Minmax_elo}"))
                else: 
                    logging.warning(('=' * 20, " {}  Game End  ".format(i + 1), '=' * 20))
                    logging.warning((f"Player_win: {Player1_win}, enemy_win: {Player2_win}, Draw: {draw}"))
                    logging.warning((f"player1_elo: {player1_elo},Minmax_elo: {Minmax_elo}"))

                if USE_TENSORBOARD:
                    Writer1.add_scalar('elo', player1_elo, step)
                    Writer2.add_scalar('elo', Minmax_elo, step)
                evaluator.reset()
    return player1_elo,Minmax_elo,step

player1_elo = 1500
Minmax_elo = 1500

step = 0

if __name__ == '__main__':

    print('cuda:', use_cuda)
    np.set_printoptions(suppress=True)

    player1_model_folder = "./data_ranmini400"
    player1_model = os.listdir(player1_model_folder)

    player1_model_id = "ranmini1"
    player2_model_id = 'minmax'

    for index in range(0,len(player1_model)):
        player1_model_path = os.path.join(player1_model_folder, player1_model[index])
        player1_elo,player2_elo,step = main(player1_model_id,player2_model_id, player1_model_path,player1_elo,Minmax_elo, step)