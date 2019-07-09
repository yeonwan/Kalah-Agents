
import importlib
from collections import namedtuple
from itertools import count
from kalah import Kalah
from runner import Runner, Minimax
from DQN import DQN
import torch.optim as optim
from kalah import reverse_board
from runner import Player
import math
import copy
import torch.nn.functional as F
import numpy as np
import torch
import random
from graphviz import Graph
import time

steps_done = 0
INF = 99999999

time_out = 5.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 32
# BATCH_SIZE = 1000
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 50000
TARGET_UPDATE = 2


left_policy_net = DQN(6).to(device)
right_policy_net = DQN(6).to(device)
left_target_net = DQN(6).to(device)
right_target_net = DQN(6).to(device)

#right_policy_net.load_state_dict(torch.load('checkpoint_left.pth', map_location=device))  # 저장 했던 거 불러와서
#left_policy_net.load_state_dict(torch.load('checkpoint_left.pth', map_location=device),strict=False)  # 다시 학습 시키기 위해

left_target_net.load_state_dict(left_policy_net.state_dict())
right_target_net.load_state_dict(right_policy_net.state_dict())
optimizer_left = optim.Adam(left_policy_net.parameters())
optimizer_right = optim.Adam(right_policy_net.parameters())
left_target_net.eval()
right_target_net.eval()


def is_f_hole(board, i):
    last_piece = (i + board[i]) % 13
    return last_piece == 6


def is_c_hole(board, i):
    last_piece = (i + board[i]) % 13
    return last_piece < 6 and board[last_piece] == 0


def game_state_to_tensor(current_board):
    c_h = []
    f_h = []
    z_h = []

    for i in range(6):
        if is_c_hole(current_board, i): # c hole
            c_h.append(current_board[12-i])
        else:
            c_h.append(0)

        if current_board[i] == 0: # zero
            z_h.append(0)
        else:
            z_h.append(1)

        if is_f_hole(current_board, i): # f hole
            f_h.append(1)
        else :
            f_h.append(0)
    f_h.append(0)
    c_h.append(0)
    z_h.append(0)

    c = np.array([c_h] * 4 , np.int)
    f = np.array([f_h] * 4, np.int)
    z = np.array([z_h] * 4, np.int)

    my_score = current_board[6]
    oppo_score = current_board[-1]
    arr_u = np.array(current_board[0:7], np.int)
    arr = np.append(arr_u, arr_u)
    arr_o = np.array(current_board[7:14], np.int)
    arr = np.append(arr, arr_o)
    arr = np.append(arr, arr_o)
    arr = np.append(arr, arr)
    arr_u = np.array([arr_u] * 4, np.int)
    arr_o = np.array([arr_o] * 4, np.int)
    arr = np.append(arr, arr_u)
    arr = np.append(arr, arr_o)

    arr = np.append(arr, np.array([0] * 28, np.int))
    arr = np.append(arr, np.array([int(my_score)] * 28, np.int))
    arr = np.append(arr, np.array([int(my_score)] * 28, np.int))
    arr = np.append(arr, np.array([int(my_score)] * 28, np.int))
    arr = np.append(arr, np.array([0] * 28, np.int))
    arr = np.append(arr, np.array([int(oppo_score)] * 28, np.int))
    arr = np.append(arr, np.array([int(oppo_score)] * 28, np.int))
    arr = np.append(arr, np.array([int(oppo_score)] * 28, np.int))
    for i in range(8):
        arr = np.append(arr, c)
    arr = np.append(arr, np.array([0] * 28, np.int))
    for i in range(8):
        arr = np.append(arr, f)
    arr = np.append(arr, np.array([0] * 28, np.int))
    for i in range(2):
        arr = np.append(arr, z)

    arr = arr.reshape(32, 4, 7)

    ret = torch.from_numpy(arr)
    ret = ret.to(device=device, dtype=torch.float).unsqueeze(0)
    ret = F.pad(input=ret, pad=(1, 1, 1, 1), mode='constant', value=0) # 12(dimension) x 6 x 9
    #print(ret)
    return ret


class Tree:
    def __init__(self, board, node, edge, node_id, position=None, is_my_move=True, parent=None, is_game_over=False):
        self.child = [None] * 6  # the children nodes corresponding to nodes after picking 1st~6th holes each.
        self.parent = parent  # the parent node
        self.position = position  # the position chose in previous step. If self is not the root node,
        # self.parent.child[self.position] is self
        if self.position is not None:
            assert 6 > self.position >= 0
        self.is_my_move = is_my_move
        self.board = board
        self.is_game_over = is_game_over

        self.cumulative_reward = 0.  # cumulated reward of children and itself
        self.n = 0  # the number of times the child node was selected

        self.node_id = node_id
        self.node = node
        self.edge = edge

    def has_child(self):
        return bool(sum([int(c is not None) for c in self.child]))

    def has_child_full(self):
        return sum([int(c is not None) for c in self.child]) == 6

    def expected_winning_rate(self):
        return self.cumulative_reward / self.n

    def UCB(self, N):
        winning_rate = self.expected_winning_rate() if self.is_my_move else 1 - self.expected_winning_rate()
        return winning_rate + math.sqrt(2 * math.log(N) / self.n)


class User(Player):
    def __init__(self, simulation_depth=6, number_of_simulation=1000, policy=None):
        super().__init__()
        '''
        Check Player class in runner.py to get information about useful predefined functions
        e.g. move, get_score, is_empty, step
        '''
        self.root = None
        self.alpha = 0.5
        self.simulation_depth = simulation_depth
        self.number_of_simulation = number_of_simulation
        self.g = Graph(format='png', graph_attr={}, node_attr={'fontsize': '13'})  # visualization graph
        self.node_id = 0
        self.policy = policy

    def build_node(self, board, position=None, is_my_move=True, parent=None, is_game_over=False):
        node_id = str(self.node_id)
        self.node_id += 1

        # visualization
        if parent is None:  # build root node
            label = 'root'
            self.g.node(node_id, label=label, shape='circle', width='0.1')
            node = self.g.body[-1]
            edge = None
        else:  # expansion
            label = str(position) if parent.parent is None else ''
            shape = 'circle' if is_my_move else 'square'
            self.g.node(node_id, label=label, shape=shape, width='0.1')
            node = self.g.body[-1]
            self.g.edge(parent.node_id, node_id)
            edge = self.g.body[-1]
            if parent.parent is None:
                '''
                Sort the children
                '''
                pos_index = sorted([(c.position, c.node_id, c.node) for c in parent.child if c is not None] + [
                    (position, node_id, node)], key=lambda c: c[0], reverse=True)
                graph_index = sorted(pos_index, key=lambda c: c[1], reverse=True)
                for gi in graph_index:
                    del self.g.body[self.g.body.index(gi[2])]
                for pi in pos_index:
                    self.g.body.insert(0, pi[2])

        return Tree(board, node, edge, node_id, position, is_my_move=is_my_move, parent=parent,
                    is_game_over=is_game_over)

    def recycle_tree(self, node):
        '''
        Recycle the past tree when updating the root for visualization
        '''
        self.g.body.append(node.node)
        if node.edge is not None:
            self.g.body.append(node.edge)
            self.g.node(node.node_id, style='filled')

        if not node.has_child():
            return
        for c in node.child:
            if c is None:
                continue
            self.recycle_tree(c)

    def initial_root(self, board):
        self.root = self.build_node(board)

    def update_root(self, position, board, is_my_move):
        '''
        Update the root as the selected child node
        '''
        del self.g
        self.g = Graph(format='png', graph_attr={'fixed_size': 'false'}, node_attr={'fontsize': '13'})

        if self.root.child[position] is None or self.root.child[position].is_my_move != is_my_move:
            self.root = self.build_node(board)
        else:
            self.root = self.root.child[position]
            del self.root.parent
            self.root.parent = None
            # change name in graph
            self.root.node = self.root.node.split("label=" + str(self.root.position))[0] + "label=" + str('root') + \
                             self.root.node.split("label=" + str(self.root.position))[1]
            self.root.edge = None
            for c in self.root.child:
                if c is not None:
                    c.node = c.node.split("label=\"\"")[0] + "label=" + str(c.position) + c.node.split("label=\"\"")[1]
        self.recycle_tree(self.root)

    def search(self, board, policy):
        start_time = time.time()
        # 이 시뮬레이션 횟수만큼 다 돌리면
        for _ in range(self.number_of_simulation):
            # assert self.root.is_my_move
            # if time.time() - start_time >= time_out:
            #     break
            # 내 트리 정책에 따라 노드를 고른다.
            # 지금 루트 노드의 자식 중 아직 None이 있으면 그냥 자기 자신을 고르고
            # None이 없으면 다 방문한것이기 때문에 그놈들 중에 Best 골라서 걔를 루트로 바꿔준다.
            selected_node = self.tree_policy()
            reward = self.simulation(selected_node.board, selected_node.is_my_move, policy, 0)
            self.backpropagation(selected_node, reward)

        # 루트 노드의 자식들에게는 위닝 레이트를 계산할 수 있게 되고, 그중 최고를 골라서 그 최고의 인덱스를 골라 리턴한다.
        # choose the action that leads to the highest expected winning rate
        expected = []
        for c in self.root.child:
            if c is None:
                expected.append(-INF)
            else:
                expected.append(c.expected_winning_rate())
        decision = self.root.child[expected.index(max(expected))]

        return decision.position

    #############################################
    # ----- MCTS algorithm step 1&2, 3, 4 ----- #
    #############################################
    def tree_policy(self):
        # Not implemented
        '''
        Do (1)Selection and (2)Expansion step
        return: the node selected for simulation
        '''


        # (1)Selection
        node = self.root  # start from root node
        if node.is_game_over:  # root node is game over
            return node  # cannot perform expansion
        node = self.best_action(node)
        # (2)Expansion
        is_my_move = node.is_my_move
        for i in range(6):
            if node.child[i] is None:
                board_i, over_i, free_turn = self.step(i, node.board, is_my_move)
                if board_i is None:
                    continue
                # (2-1)check free_turn
                next_is_my_move = is_my_move if free_turn else not is_my_move
                # (2-2)make new node
                node.child[i] = self.build_node(board_i, i, next_is_my_move, node, over_i)
                return node.child[i]
        # if node's child is already fully generated,
        # Then best_action may return best UCB child. return it.
        return node

    def simulation(self, board, is_my_move, policy, depth=0):

        if self.is_game_over(board):
            return self.evaluation_v1(board, policy)

        action, cur_board, is_my = self.default_policy(board, is_my_move, policy)
        board_next, over_next, free_turn = self.step(action, cur_board, is_my)

        if board_next is None:
            print("BUG")
        next_is_my_move = is_my_move if free_turn else not is_my_move
        # base case : simulation 은 게임이 끝날때까지 진행한다. 또는 depth limit에 걸릴 경우.
        if over_next or depth == self.simulation_depth:
            return self.evaluation_v1(board_next, policy)
        return self.simulation(board_next,  next_is_my_move, policy, depth+1)

    def backpropagation(self, node, reward):
        while node is not None:
            node.n += 1
            node.cumulative_reward += reward
            if node.parent is not None:
                self.g.node(node.node_id, style='filled')
            node = node.parent

    #############################################
    # -------- subfunctions for MCTS ---------- #
    #############################################

    def best_action(self, _node):
        # Not implemented
        max_ucb = 0
        max_idx = 0
        node = _node
        while node.has_child_full():
            for idx, child in enumerate(node.child):
                ucb = child.UCB(node.n)  # UCB에 넣는 N param 잘 계산했는지 나중에 체크해보아야 함
                if ucb > max_ucb:
                    max_ucb = ucb
                    max_idx = idx
            node = node.child[max_idx]
        return node

    def default_policy(self, board, is_my_move, policy):

        board = copy.deepcopy(board)
        if not is_my_move:
            target_board = reverse_board(board)
        else:
            target_board = board

        sample = random. random()

        if sample > 0.96: # 가끔 default policy 에 model 로 결정.
            state = game_state_to_tensor(target_board)
            actions = policy(state).squeeze(0).tolist()
            max = -100
            max_idx = 0
            for idx, action in enumerate(actions):
                if target_board[idx] != 0 and action > max:
                    max = action
                    max_idx = idx
            return max_idx, list(board), is_my_move

        else: # 아니면 랜덤
            while True:
                position = random.choice(range(6))
                if not self.is_empty(position, board, is_my_move):
                    break

            return position, list(board), is_my_move

    def evaluation_v1(self, board, policy):
        '''
        Evaluate when game is over or reaches the last depth of simulation
        You can change this rewards to improve the performance
        '''
        user_score = board[6]
        oppo_score = board[-1]
        user_pieces = sum(board[:6])
        oppo_pieces = sum(board[7:-1])
        if self.is_game_over(board):  # win: 1, dual: 0.5, lose: 0
            if user_score + user_pieces > oppo_score + oppo_pieces:
                reward = 1
            elif user_score + user_pieces == oppo_score + oppo_pieces:
                reward = 0.5
            else:
                reward = 0
        else:
            state = game_state_to_tensor(board)
            reward = policy(state)
            #print(reward)
            reward = reward.max(1)[0].view(1, 1).item()

        return reward

    #############################################
    # ----------- other functions ------------- #
    #############################################

    def is_game_over(self, board):
        if sum(board[:6]) == 0 or sum(board[7:-1]) == 0:
            return True
        else:
            return False

    def print_winning_rate(self, next_position):
        if self.root.child[next_position] is None or self.root.child[next_position].n == 0:
            return None
        else:
            return self.root.child[next_position].cumulative_reward / self.root.child[next_position].n


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """전환 저장"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


memory = ReplayMemory(10000)
runner = Runner(1, user_path='player_wan.User', opponent_path=None)


def select_action(cur_board, policy):

    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-0.3 * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold: # 지수적으로 MCST search 를 더 많이하도록
        return torch.tensor([[runner.user.search(cur_board, policy)]], device=device, dtype=torch.long) # MCTS Search
    else:
        return torch.tensor([[random.randrange(6)]], device=device, dtype=torch.long) # 랜더


def optimize_model(free_turn, policy, target, optimizer):
    if len(memory) < BATCH_SIZE:
        return 0
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    # 최종 상태가 아닌 마스크를 계산하고 배치 요소를 연결합니다.
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])  # 이어붙이기
    state_batch = torch.cat(batch.state)  # 이어붙이기
    action_batch = torch.cat(batch.action)  # 이어붙이기
    reward_batch = torch.cat(batch.reward)  # 이어붙이기

    # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 칼럼을 선택한다.
    state_action_values = policy(state_batch).gather(1, action_batch)

    # 모든 다음 상태를 위한 V(s_{t+1}) 계산
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target(non_final_next_states).max(1)[0].detach()

    # 기대 Q 값 계산1
    if free_turn:
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    else:
        expected_state_action_values = ((1 - next_state_values) * GAMMA) + reward_batch

    # Huber 손실 계산
    loss = F.l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # 모델 최적화
    optimizer.zero_grad()
    loss.backward()
    for param in policy.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item()


def evaluation(is_game_over, board):
    if is_game_over:  # win: 1, dual: 0.5, lose: 0
        user_score = board[6]
        oppo_score = board[-1]
        user_pieces = sum(board[:6])
        oppo_pieces = sum(board[7:-1])
        if user_score + user_pieces > oppo_score + oppo_pieces:
            reward = 1
        elif user_score + user_pieces == oppo_score + oppo_pieces:
            reward = 0.5
        else:
            reward = 0
    else:
        reward = 0
    return reward


def main():
    num_episodes = 10000
    # player = User(simulation_depth=6, number_of_simulation=1000)

    if runner.am_i_minmax:
        runner.user = Minimax()
    else:
        runner.user = User(simulation_depth=6, number_of_simulation=200)

    if runner.is_user_defined_opponent:
        module, name = runner.opponent_path.rsplit('.', 1)
        runner.opponent = getattr(importlib.import_module(module), name)(number_of_simulation=1000)
    else:
        runner.opponent = Minimax()

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        print("New games for training!")
        initial_board = [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0]
        new_game = Kalah(initial_board)
        if i_episode % 2 == 0:
            new_game.player = False

        if not runner.am_i_minmax:
            runner.user.initial_root(initial_board)

        if runner.is_user_defined_opponent:
            runner.opponent.initial_root(initial_board)
        num = 0
        loss_sum = 0
        for turn in count():
            # 행동 선택과 수행

            current_board = copy.deepcopy(new_game.get_board())
            state = game_state_to_tensor(current_board)

            if new_game.player: # 내차례
                cur_policy = left_policy_net # 왼쪽 모델
                cur_target = left_target_net
                opt = optimizer_left
                while True:
                    action = select_action(current_board, cur_target)
                    next_position = action.item()
                    if new_game.get_board()[next_position] != 0 or new_game.is_game_over():
                        break

            else: # 적 차례
                cur_policy = left_policy_net # 오른쪽 모델
                cur_target = left_target_net
                opt = optimizer_left
                while True:
                    #action = torch.tensor([[runner.opponent.search(current_board)]], device=device)
                    action = select_action(current_board, cur_target)
                    next_position = action.item()
                    if new_game.get_board()[next_position] != 0 or new_game.is_game_over():
                        break

            _, free_turn = new_game.move(next_position)

            # 새로운 상태 관찰
            next_board = copy.deepcopy(new_game.get_board())
            next_state = game_state_to_tensor(next_board)
            reward = evaluation(new_game.is_game_over(), next_board)
            # 메모리에 변이 저장
            reward = torch.tensor([reward], device=device, dtype=torch.float)
            memory.push(state, action, next_state, reward)
            # 로스 계산
            loss = optimize_model(free_turn, cur_policy, cur_target, opt)
            loss_sum += loss

            if not runner.am_i_minmax:
                runner.user.update_root(next_position, copy.deepcopy(new_game.get_board()), copy.deepcopy(new_game.player))

            if runner.is_user_defined_opponent:
                runner.opponent.update_root(next_position, copy.deepcopy(new_game.get_board()),
                                            copy.deepcopy(not new_game.player))
            if new_game.is_game_over():
                num = turn
                break

        runner.score_board(i_episode, new_game.result())
        print(i_episode, 'game Average loss: ', loss_sum/num)
        print()
        # 목표 네트워크 업데이트
        if i_episode % TARGET_UPDATE == 1:
            left_target_net.load_state_dict(left_policy_net.state_dict())
            #right_target_net.load_state_dict(right_policy_net.state_dict())

        if i_episode % 50 == 49: # 50번 마다 한번 저장
            torch.save(left_target_net.state_dict(), 'checkpoint_left.pth')
            runner.wins = 0
            runner.losses = 0
            runner.draws = 0

            #torch.save(right_target_net.state_dict(), 'checkpoint_right.pth')


    torch.save(left_target_net.state_dict(), 'dqn_cnn_left.pth')
    #torch.save(right_target_net.state_dict(), 'dqn_cnn_right.pth')
    print('Complete')


if __name__ == '__main__':
    main()