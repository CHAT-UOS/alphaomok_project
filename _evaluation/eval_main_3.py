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
    Writer1 = SummaryWriter("elo/ranmini1")
    Writer2 = SummaryWriter("elo/mini1")
    Writer3 = SummaryWriter("elo/base")

BOARD_SIZE = game.Return_BoardParams()[0]

N_BLOCKS_PLAYER1 = 5
N_BLOCKS_PLAYER2 = 5
N_BLOCKS_PLAYER3 = 5

IN_PLANES_PLAYER1 = 5  
IN_PLANES_PLAYER2 = 5
IN_PLANES_PLAYER3 = 5

OUT_PLANES_PLAYER1 = 128
OUT_PLANES_PLAYER2 = 128
OUT_PLANES_PLAYER3 = 128

N_MCTS_PLAYER1 = 100
N_MCTS_PLAYER2 = 100
N_MCTS_PLAYER3 = 400

N_MATCH = 100

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

class Evaluator(object):
    def __init__(self):
        self.player1 = None
        self.player2 = None
        self.player3 = None
        pass

    def set_agents(self, black,white,black_id,white_id):
        
        game_mode = 'text'

        self.env = game.GameState(game_mode)

        print(f'load {black_id} model:', black,"\n")
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

        if PRINT_SELFPLAY == True:
            print(f'load {white_id} model:', white,"\n")

        self.player2 = agents.ZeroAgent(BOARD_SIZE,
                                        N_MCTS_PLAYER2,
                                        IN_PLANES_PLAYER2,
                                        noise=False)
            
        self.player2.model = model.PVNet(N_BLOCKS_PLAYER2,
                                    IN_PLANES_PLAYER2,
                                    OUT_PLANES_PLAYER2,
                                    BOARD_SIZE).to(device)

        state_b = self.player2.model.state_dict()
        my_state_b = torch.load(
            white, map_location='cuda:0' if use_cuda else 'cpu')
        for k, v in my_state_b.items():
            if k in state_b:
                state_b[k] = v
        self.player2.model.load_state_dict(state_b)

    def get_action(self, root_id, board, turn, enemy_turn,time_step):
        if turn != enemy_turn:
            if time_step <= 2:
                pi = self.player1.get_random_pi(root_id)
            else:
                pi = self.player1.get_pi(root_id, tau=0)
        else:
            if time_step <= 2:
                pi = self.player2.get_random_pi(root_id)
            else:
                pi = self.player2.get_pi(root_id, tau=0)

        action, action_index = utils.argmax_onehot(pi)

        return action, action_index

    def return_env(self):
        return self.env

    def reset(self):
        self.player1.reset()
        self.player2.reset()

    def put_action(self, action_idx, turn, enemy_turn):
        
        if PRINT_SELFPLAY == True:
            print(self.player1)

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

def main(player1_model_id,player2_model_id,player3_model_id,player1_model_path,player2_model_path,player3_model_path,player1_elo,player2_elo,player3_elo,step):
    turn = 0
    enemy_turn = 1

    elo1 = "player1_elo"
    elo2 = "player2_elo"
    elo3 = "player3_elo"

    play_order = [[player1_model_path,player2_model_path,player1_model_id,player2_model_id,elo1,elo2],
                   [player2_model_path,player3_model_path,player2_model_id,player3_model_id,elo2,elo3],
                   [player3_model_path,player1_model_path,player3_model_id,player1_model_id,elo3,elo1]]

    for match in play_order:

        black = match[0]
        white = match[1]

        black_id = match[2]
        white_id = match[3]

        black_elo = match[4]
        white_elo = match[5]

        Player1_win = 0
        Player2_win = 0
        draw = 0
        
        evaluator.set_agents(black, white,black_id,white_id)
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
                            if black_elo == elo1 and white_elo == elo2:
                                player1_elo, player2_elo = elo(
                                    player1_elo, player2_elo, 0.5, 0.5)
                            elif black_elo == elo2 and white_elo == elo3:
                                player2_elo, player3_elo = elo(
                                    player2_elo, player3_elo, 0.5, 0.5)
                            elif black_elo == elo3 and white_elo == elo1:
                                player3_elo, player1_elo = elo(
                                    player3_elo, player1_elo, 0.5, 0.5)

                        else:
                            print(f'\n{black_id} Win!')
                            Player1_win += 1
                            if black_elo == elo1 and white_elo == elo2:
                                player1_elo, player2_elo = elo(
                                    player1_elo, player2_elo, 1, 0)
                            elif black_elo == elo2 and white_elo == elo3:
                                player2_elo, player3_elo = elo(
                                    player2_elo, player3_elo, 1, 0)
                            elif black_elo == elo3 and white_elo == elo1:
                                player3_elo, player1_elo = elo(
                                    player3_elo, player1_elo, 1, 0)

                    else:
                        if win_index == 3:
                            print('\nDraw!')
                            draw +=1
                            if black_elo == elo1 and white_elo == elo2:
                                player1_elo, player2_elo = elo(
                                    player1_elo, player2_elo, 0.5, 0.5)
                            elif black_elo == elo2 and white_elo == elo3:
                                player2_elo, player3_elo = elo(
                                    player2_elo, player3_elo, 0.5, 0.5)
                            elif black_elo == elo3 and white_elo == elo1:
                                player3_elo, player1_elo = elo(
                                    player3_elo, player1_elo, 0.5, 0.5)
                        else:
                            print(f'\n{white_id} Win!')
                            Player2_win += 1
                            if black_elo == elo1 and white_elo == elo2:
                                player1_elo, player2_elo = elo(
                                    player1_elo, player2_elo, 0, 1)
                            elif black_elo == elo2 and white_elo == elo3:
                                player2_elo, player3_elo = elo(
                                    player2_elo, player3_elo, 0, 1)
                            elif black_elo == elo3 and white_elo == elo1:
                                player3_elo, player1_elo = elo(
                                    player3_elo, player1_elo, 0, 1)
                                
                    utils.render_str(board, BOARD_SIZE, action_index)
                    # Change turn
                    enemy_turn = abs(enemy_turn - 1)
                    turn = 0
                    if PRINT_SELFPLAY == True:
                        print(('=' * 20, " {}  Game End  ".format(i + 1), '=' * 20))
                        print((f"Player_win: {Player1_win}, enemy_win: {Player2_win}, Draw: {draw}"))
                        print((f"player1_elo: {player1_elo},player2_elo: {player2_elo},player3_elo: {player3_elo}"))

                    else: 
                        logging.warning('=' * 20, " {}  Game End  ".format(i + 1), '=' * 20)
                        logging.warning(f"Player_win: {Player1_win}, enemy_win: {Player2_win}, Draw: {draw}")
                        logging.warning(f"player1_elo: {player1_elo},player2_elo: {player2_elo},player3_elo: {player3_elo}")

                    if USE_TENSORBOARD:
                        Writer1.add_scalar('elo', player1_elo, step)
                        Writer2.add_scalar('elo', player2_elo, step)
                        Writer3.add_scalar('elo', player3_elo, step)
                    evaluator.reset()
    return player1_elo,player2_elo,player3_elo,step

player1_elo = 1500
player2_elo = 1500
player3_elo = 1500

step = 0

# Web API
if __name__ == '__main__':

    print('cuda:', use_cuda)
    np.set_printoptions(suppress=True)

    player1_model_folder = "./data1"
    player2_model_folder = "./data2"
    player3_model_folder = "./data3"

    player1_model = os.listdir(player1_model_folder)
    player2_model = os.listdir(player2_model_folder)
    player3_model = os.listdir(player3_model_folder)

    player1_model_id = 'ranmini1'
    player2_model_id = 'base_minification1'
    player3_model_id = 'base'

    if len(player1_model) != len(player2_model) != player3_model_id:
        sys.exit("Please check the number of models in each 'data' folder.")

    for index in range(0,len(player1_model)):
        player1_model_path = os.path.join(player1_model_folder, player1_model[index])
        player2_model_path = os.path.join(player2_model_folder, player2_model[index])
        player3_model_path = os.path.join(player3_model_folder, player3_model[index])
        player1_elo,player2_elo,player3_elo,step = main(player1_model_id,player2_model_id,player3_model_id,player1_model_path,player2_model_path,player3_model_path,player1_elo,player2_elo,player3_elo,step)