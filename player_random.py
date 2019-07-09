from kalah import reverse_board
from runner import Player
import math
import random
import copy
from graphviz import Graph
import seaborn
import time

INF = 99999999
color_hex = seaborn.color_palette('RdBu_r', 101).as_hex()
time_out = 3.


def get_color_hex(winning_rate):
    # 0 <= winning_rate <= 1
    assert 0. <= winning_rate <= 1.
    return color_hex[int(winning_rate * 100)]


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
    def __init__(self, simulation_depth=6, number_of_simulation=1000):
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
            self.g.node(node.node_id, style='filled', color=get_color_hex(node.expected_winning_rate()))

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

    def search(self, board):
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
            reward = self.simulation(selected_node.board, selected_node.is_my_move)
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

    def simulation(self, board, is_my_move):
        board_next, over_next, free_turn = self.default_policy(board, is_my_move)
        next_is_my_move = is_my_move if free_turn else not is_my_move
        if over_next:  # base case : simulation 은 게임이 끝날때까지 진행한다.
            return self.evaluation(board_next, next_is_my_move)
        return self.simulation(board_next, next_is_my_move)

    def backpropagation(self, node, reward):
        while node is not None:
            node.n += 1
            node.cumulative_reward += reward
            if node.parent is not None:
                self.g.node(node.node_id, style='filled', color=get_color_hex(node.expected_winning_rate()))
            node = node.parent

    #############################################
    # -------- subfunctions for MCTS ---------- #
    #############################################

    def best_action(self, node):
        # Not implemented
        '''
        Case 1: if there exists unvisited child
            return unvisited
        Case 2: else
            return the child who has the maximum UCB
        '''

        max_ucb = 0
        max_idx = 0

        if not node.has_child_full():  # 모두 방문했는지 체크하는 로직 나중에 체크해보아야 함
            return node
        else:  # child 6개 모두 방문했으면 best child를 UCB로 계산하기
            for idx, child in enumerate(node.child):
                ucb = child.UCB(node.n)  # UCB에 넣는 N param 잘 계산했는지 나중에 체크해보아야 함
                if ucb > max_ucb:
                    max_ucb = ucb
                    max_idx = idx
            return node.child[max_idx]

    def default_policy(self, board, is_my_move):
        '''
        Randomly select non-empty position
        '''
        position = 0
        while True:
            position = random.choice(range(6))
            if not self.is_empty(position, board, is_my_move):
                break

        return self.step(position, list(board), is_my_move)

    def evaluation_v1(self, board, is_my_move):
        user_score = board[6]
        oppo_score = board[-1]

        if self.is_game_over(board):  # win: 1, dual: 0.5, lose: 0
            user_pieces = sum(board[:6])
            oppo_pieces = sum(board[7:-1])
            if user_score + user_pieces > oppo_score + oppo_pieces:
                reward = 1
            elif user_score + user_pieces == oppo_score + oppo_pieces:
                reward = 0.5
            else:
                reward = 0
        else:
            oppo_board = reverse_board(board)
            for i in range(0, 6):
                user_piece = board[i]
                oppo_idx = i + 7
                oppo_piece = oppo_board[oppo_idx]
                if self.is_f_hole(board, i):
                    user_score += user_piece
                elif self.is_c_hole(board, i):
                    user_score += user_piece
                else:
                    user_score += user_piece * (0.1 + i * 0.1)

                if self.is_f_hole(oppo_board, oppo_idx) or self.is_c_hole(oppo_board, oppo_idx):
                    oppo_score += oppo_piece
                else:
                    oppo_score += oppo_piece * (0.1 + i * 0.1)
            reward = user_score / (user_score + oppo_score)

        return reward

    def evaluation(self, board, is_my_move):
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
            user_score += user_pieces * self.alpha
            oppo_score += oppo_pieces * self.alpha
            reward = user_score / (user_score + oppo_score)

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
