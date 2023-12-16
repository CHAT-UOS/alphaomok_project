import logging
import pickle
import random
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import model
import utils
import agents

# env_small: 9x9, env_regular: 15x15
from env import env_small as game

logging.basicConfig(
    filename='logs/log_'+datetime.now().strftime('%y%m%d')+'.txt',
    level=logging.WARNING)
# Game
BOARD_SIZE = game.Return_BoardParams()[0] # 9 or 15
N_MCTS = 100
TAU_THRES = 6
SEED = 0
PRINT_SELFPLAY = False

# Net
N_BLOCKS = 5
IN_PLANES = 5  # history * 2 + 1
OUT_PLANES = 128

# Training
USE_TENSORBOARD = True
N_SELFPLAY = 100
TOTAL_ITER = 10000000
MEMORY_SIZE = 30000
N_EPOCHS = 1
BATCH_SIZE = 32
LR = 2e-4
L2 = 0

# Hyperparameter sharing
agents.PRINT_MCTS = PRINT_SELFPLAY

# Set gpu or cpu
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('cuda:', use_cuda)

# Numpy printing style
np.set_printoptions(suppress=True)

# Set random seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if use_cuda:
    torch.cuda.manual_seed_all(SEED)

# Global variables
rep_memory = deque(maxlen=MEMORY_SIZE) # 30000
cur_memory = deque() #deque: python 내장 모듈인 collections 에서 제공되는 자료 구조로 양쪽 끝에서 데이터를 효율적으로 추가하고 삭제할 수 있는 자료 구조 
step = 0
start_iter = 0
total_epoch = 0
result = {'Black': 0, 'White': 0, 'Draw': 0}
if USE_TENSORBOARD:
    from tensorboardX import SummaryWriter
    Writer = SummaryWriter("../visualization/1125/ranmini1")

Agent = agents.ZeroAgent(BOARD_SIZE,
                         N_MCTS,
                         IN_PLANES,
                         noise=True)
Agent.model = model.PVNet(N_BLOCKS,
                          IN_PLANES,
                          OUT_PLANES,
                          BOARD_SIZE).to(device)

optimizer = optim.Adam(Agent.model.parameters(), lr=LR, weight_decay=L2, eps=1e-6)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
print(get_n_params(Agent.model))

logging.warning(f'\nCUDA: {use_cuda}\
    \nAGENT: {type(Agent).__name__}\
    \nMODEL: {type(Agent.model).__name__}\
    \nSEED: {SEED}\
    \nBOARD_SIZE: {BOARD_SIZE}\
    \nN_MCTS: {N_MCTS}\
    \nTAU_THRES: {TAU_THRES}\
    \nN_BLOCKS: {N_BLOCKS}\
    \nIN_PLANES: {IN_PLANES}\
    \nOUT_PLANES: {OUT_PLANES}\
    \nN_SELFPLAY: {N_SELFPLAY}\
    \nMEMORY_SIZE: {MEMORY_SIZE}\
    \nN_EPOCHS: {N_EPOCHS}\
    \nBATCH_SIZE: {BATCH_SIZE}\
    \nLR: {LR}\
    \nL2: {L2}')

def self_play(n_selfplay):
    global cur_memory, rep_memory, Agent

    Agent.model.eval()
    state_black = deque()
    state_white = deque()
    pi_black = deque()
    pi_white = deque()

    for episode in range(n_selfplay):
        if (episode + 1) % 10 == 0:
            logging.warning(f'Playing Episode {episode+1}') #episode 10 20 30~ 마다 logging warning

        env = game.GameState('text')
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), 'float')
        turn = 0
        root_id = (0,) # 0이라는 하나의 요소를 가진 튜플 
        win_index = 0
        time_steps = 0
        action_index = None

        while win_index == 0:
            if PRINT_SELFPLAY:
                utils.render_str(board, BOARD_SIZE, action_index)

            # ====================== start MCTS ============================ #

            if time_steps < TAU_THRES:
                tau = 1
            else:
                tau = 0

            pi = Agent.get_pi(root_id, tau) # 

            # ===================== collect samples ======================== #

            state = utils.get_state_pt(root_id, BOARD_SIZE, IN_PLANES)

            if turn == 0:
                state_black.appendleft(state)
                pi_black.appendleft(pi)
            else:
                state_white.appendleft(state)
                pi_white.appendleft(pi)

            # ======================== get action ========================== #

            action, action_index = utils.get_action(pi)
            root_id += (action_index,)

            # ====================== print evaluation ====================== #

            if PRINT_SELFPLAY:
                with torch.no_grad():
                    state_input = torch.tensor([state]).to(device).float()
                    p, v = Agent.model(state_input)
                    p = p.cpu().numpy()[0]
                    v = v.item()

                    print('\nPi:\n{}'.format(
                        pi.reshape(BOARD_SIZE, BOARD_SIZE).round(decimals=2)
                    ))
                    print('\nPolicy:\n{}'.format(
                        p.reshape(BOARD_SIZE, BOARD_SIZE).round(decimals=2)
                    ))

                if turn == 0:
                    print(f"\nBlack's win%: {(v + 1) / 2 * 100:.2f}%")
                else:
                    print(f"\nWhite's win%: {(v + 1) / 2 * 100:.2f}%")

            # =========================== step ============================= #

            board, _, win_index, turn, _ = env.step(action)
            time_steps += 1 # 흑 or 백돌이 놓일 때마다 time_Steps +1

            # ========================== result ============================ #

            if win_index != 0:
                if win_index == 1:
                    reward_black = 1.
                    reward_white = -1.
                    result['Black'] += 1

                elif win_index == 2:
                    reward_black = -1.
                    reward_white = 1.
                    result['White'] += 1

                else:
                    reward_black = 0.
                    reward_white = 0.
                    result['Draw'] += 1

            # ====================== store in memory ======================= #

                while state_black or state_white:
                    if state_black:
                        cur_memory.append((state_black.pop(),
                                           pi_black.pop(),
                                           reward_black))
                    if state_white:
                        cur_memory.append((state_white.pop(),
                                           pi_white.pop(),
                                           reward_white))

            # =========================  result  =========================== #

                if PRINT_SELFPLAY:
                    utils.render_str(board, BOARD_SIZE, action_index)

                    bw, ww, dr = result['Black'], result['White'], \
                        result['Draw']
                    print('')
                    print('=' * 20,
                          " {:3} Game End   ".format(episode + 1),
                          '=' * 20)
                    print(f'Black Win: {bw:3}\
                          White Win: {ww:3}\
                          Draw: {dr:2}\
                          Win%: {(bw + 0.5 * dr) / (bw + ww + dr) * 100:.2f}%')
                    print('current memory size:', len(cur_memory))

                Agent.reset()

    rep_memory.extend(utils.augment_dataset(cur_memory, BOARD_SIZE))


def train(n_epochs, n_iter):
    global step, total_epoch
    global Agent, optimizer, Writer
    global rep_memory, cur_memory

    Agent.model.train()
    loss_all = []
    loss_v = []
    loss_p = []
    train_memory = []
    train_memory.extend(
        random.sample(rep_memory, BATCH_SIZE * len(cur_memory))) #rep_memory에서 크기가 BATCH_SIZE * len(cur_memory)인 무작위 샘플을 추출해 train_memory에 추가 

    dataloader = DataLoader(train_memory,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            pin_memory=use_cuda)

    print('=' * 58)
    print(' ' * 20 + ' Start Learning ' + ' ' * 20)
    print('=' * 58)
    print('current memory size:', len(cur_memory))
    print('replay memory size:', len(rep_memory))
    print('train memory size:', len(train_memory))
    print('optimizer: {}'.format(optimizer))
    logging.warning('=' * 58)
    logging.warning(' ' * 20 + ' Start Learning ' + ' ' * 20)
    logging.warning('=' * 58)
    logging.warning(f'current memory size: {len(cur_memory)}')
    logging.warning(f'replay memory size: {len(rep_memory)}')
    logging.warning(f'train memory size: {len(train_memory)}')
    logging.warning(f'optimizer: {optimizer}')

    for epoch in range(n_epochs):
        for i, (s, pi, z) in enumerate(dataloader):
            s_batch = s.to(device).float()
            pi_batch = pi.to(device).float()
            z_batch = z.to(device).float()

            p_batch, v_batch = Agent.model(s_batch)

            v_loss = (v_batch - z_batch).pow(2).mean()
            p_loss = -(pi_batch * p_batch.log()).sum(dim=-1).mean()
            loss = v_loss + p_loss

            loss_v.append(v_loss.item())
            loss_p.append(p_loss.item())
            loss_all.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            if USE_TENSORBOARD:
                Writer.add_scalar('Loss', loss.item(), step)
                Writer.add_scalar('Loss_V', v_loss.item(), step)
                Writer.add_scalar('Loss_P', p_loss.item(), step)

            if PRINT_SELFPLAY:
                print('{:4} Step Loss: {:.4f}   '
                      'Loss V: {:.4f}   '
                      'Loss P: {:.4f}'.format(step,
                                              loss.item(),
                                              v_loss.item(),
                                              p_loss.item()))
        total_epoch += 1

        if PRINT_SELFPLAY:
            print('-' * 58)
            print('{:2} Epoch Loss: {:.4f}   '
                  'Loss V: {:.4f}   '
                  'Loss P: {:.4f}'.format(total_epoch,
                                          np.mean(loss_all),
                                          np.mean(loss_v),
                                          np.mean(loss_p)))
        logging.warning('{:2} Epoch Loss: {:.4f}   '
                        'Loss_V: {:.4f}   '
                        'Loss_P: {:.4f}'.format(total_epoch,
                                                np.mean(loss_all),
                                                np.mean(loss_v),
                                                np.mean(loss_p)))

def save_model(agent, n_iter, step):
    torch.save(
        agent.model.state_dict(),
        f'data/{datetime_now}_{n_iter}_{step}_step_model.pickle')

def save_dataset(memory, n_iter, step):
    with open('data/{}_{}_{}_step_dataset.pickle'.format(
            datetime_now, n_iter, step), 'wb') as f:
        pickle.dump(memory, f, pickle.HIGHEST_PROTOCOL)


def load_data(model_path, dataset_path):
    global rep_memory, step, start_iter
    if model_path:
        print('load model: {}'.format(model_path))
        logging.warning('load model: {}'.format(model_path))
        state = Agent.model.state_dict()
        state.update(torch.load(model_path))
        Agent.model.load_state_dict(state)
        step = int(model_path.split('_')[2])
        start_iter = int(model_path.split('_')[1]) + 1
    if dataset_path:
        print(f'load dataset: {dataset_path}')
        logging.warning(f'load dataset: {dataset_path}')
        with open(dataset_path, 'rb') as f:
            rep_memory = deque(pickle.load(f), maxlen=MEMORY_SIZE)


def reset_iter(result, cur_memory):
    global total_epoch
    result['Black'] = 0
    result['White'] = 0
    result['Draw'] = 0
    total_epoch = 0
    cur_memory.clear()


if __name__ == '__main__':
    model_path = "./data/231126_5400_75223_step_model.pickle"
    dataset_path = "./data/231126_5400_75223_step_dataset.pickle"

    load_data(model_path, dataset_path)

    for n_iter in range(start_iter, TOTAL_ITER): # 0~10000000

        print(f" progress: {n_iter/TOTAL_ITER*100:.3f}%")
        print('=' * 58)
        print(' ' * 20 + f'  Iteration {n_iter}  ' + ' ' * 20)
        print('=' * 58)

        logging.warning(datetime.now().isoformat()) #  "2023-11-05T14:30:00.123456"
        logging.warning('=' * 58)
        logging.warning(' ' * 20 + f'  Iteration {n_iter}  ' + ' ' * 20)
        logging.warning('=' * 58)
        datetime_now = datetime.now().strftime('%y%m%d')

        if n_iter > 0:
            N_SELFPLAY = 1
            self_play(N_SELFPLAY) #200
            train(N_EPOCHS, n_iter)
        else:
            self_play(N_SELFPLAY) #200

        if n_iter % 400 == 0:
            save_model(Agent, n_iter, step)
            save_dataset(rep_memory, n_iter, step)
        reset_iter(result, cur_memory)
