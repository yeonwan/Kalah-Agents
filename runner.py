from kalah import Kalah, reverse_board
import copy

import inspect
import importlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

time_out = 5.01

def show_image (path, full_screen=False, auto_close=True):
    plt.close('all')
    img = mpimg.imread(path)
    plt.imshow(img, interpolation='bilinear', aspect='auto', cmap='RdBu_r')
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(img.shape[1]/float(DPI)+1, img.shape[0]/float(DPI)+2)
    plt.axis('off')
    plt.colorbar()
    if full_screen:
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
    if auto_close:
        plt.show(block=False)
        plt.pause(3)
    else:
        plt.show()
    plt.clf() #will make the plot window empty


class Player:
    def __init__(self):
        return

    def move(self, position, board, is_my_move=True):
        '''
        returns the board after given movement.
        you can get either your's or opponent's movement with the parameter "is_my_move"
        '''
        board = copy.deepcopy(board)
        prediction = Kalah(board)
        if not is_my_move:
            prediction.board = reverse_board(board)

        _, free_turn = prediction.move(position)
        board = prediction.board

        if not is_my_move:
            board = reverse_board(board)
        return board, prediction.is_game_over(), free_turn

    def get_score(self, board, is_mine=True):
        '''
        returns current score
        is_mine is True --> returns my score
        is_mine is False --> returns opponent's score
        '''
        if is_mine:
            return board[6]
        else:
            return board[-1]

    def is_empty(self, position, board, is_mine=True):
        '''
        returns whether the given position is empty or not
        '''
        if is_mine:
            return 0 >= board[position]
        else:
            return 0 >= board[position+7]

    def step(self, pos, board, is_my_move=True):
        if self.is_empty(pos, board, is_mine=is_my_move):
            return None, None, None
        new_board, over, free_turn = self.move(pos, board, is_my_move)
        return new_board, over, free_turn


class Minimax(Player):
    def __init__(self, N=6):
        super().__init__()
        self.N = N

    def loop(self, step, board, is_my_move=True):
        if step == 0 or sum([int(self.is_empty(i, board, is_my_move)) for i in range(6)]) == 6:
            return (self.get_score(board)-self.get_score(board, is_mine=False))*10+(sum(board[:6])-sum(board[7:-1]))*1, None
        step -= 1
        score = []
        position = []
        for i in range(6):

            board_i, over_i, free_turn = self.step(i, board, is_my_move)

            if board_i is None:
                continue


            next_is_my_move = is_my_move if free_turn else not is_my_move
            score_i, j = self.loop(step, board_i, next_is_my_move)
            score.append(score_i)
            position.append(i)

        return sum(score)/len(position), position[score.index(max(score))]

    def search(self, board):
        '''
        N step search
        returns the position which has the maximum score after N step
        '''
        predicted_score, next_position = self.loop(self.N, board)
        return next_position


class Runner:
    def __init__(self, num_of_games=1, user_path=None, opponent_path=None):
        self.user_path = user_path
        if user_path is None:
            self.am_i_minmax = True
        else:
            self.am_i_minmax = False

        self.opponent_path = opponent_path
        if opponent_path is None:
            self.is_user_defined_opponent = False
        else:   #load model by path
            self.is_user_defined_opponent = True
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.num_of_games = num_of_games
        self.opponent = None


    def score_board(self, ith, result):
        print(str(ith)+"th game over!")
        if result == 1:
            self.wins += 1
        elif result == 0:
            self.draws += 1
        else:
            self.losses += 1
        print("Total wins:\t"+str(self.wins))
        print("Total draws:\t"+str(self.draws))
        print("Total losses:\t"+str(self.losses))
        print("Total winning rate:\t"+str(float(self.wins / (ith+1))*100)+"%")
        return
    def is_time_out(self, start, end):
        if end - start > time_out:
            print("time out!")
            print('measured time: ', end-start, '(sec) exceeded 1 sec')
            exit(-1)

    def run_game(self, tree_visualization=True):
        for i in range(self.num_of_games):
            if self.am_i_minmax is True:
                self. user = Minimax()
            else:
                module, name = self.user_path.rsplit('.', 1)
                self.user = getattr(importlib.import_module(module), name)(number_of_simulation=500, simulation_depth = 6)

            if self.is_user_defined_opponent:
                module, name = self.opponent_path.rsplit('.', 1)
                self.opponent = getattr(importlib.import_module(module), name)(number_of_simulation=1000)
            else:
                self.opponent = Minimax()

            print("New game!")
            print("Initial board >>")
            # initialization:
            initial_board = [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0]
            new_game = Kalah(initial_board)
            if i % 2 == 1:
                new_game.player = False
            new_game.show_board()
            if not self.am_i_minmax:
                self.user.initial_root(initial_board)
            if self.is_user_defined_opponent:
                self.opponent.initial_root(initial_board)
            turn = 0

            while not new_game.is_game_over():
                turn +=1
                # pick a hole:
                if new_game.player:
                    start_time = time.time()
                    next_position = self.user.search(copy.deepcopy(new_game.get_board()))
                    end_time = time.time()
                    print('measured time: ', end_time-start_time)
                    self.is_time_out(start_time, end_time)
                else:
                    next_position = self.opponent.search(copy.deepcopy(new_game.get_board()))
                # update:

                tmp_score, free_turn = new_game.move(next_position)
                # print:
                if not self.am_i_minmax:
                    print("winning rate:", self.user.print_winning_rate(next_position))
                if tree_visualization:
                    show_image(self.user.g.render(view=False), auto_close=False)
                if not self.am_i_minmax:
                    self.user.update_root(next_position, copy.deepcopy(new_game.get_board()), copy.deepcopy(new_game.player))
                if self.is_user_defined_opponent:
                    self.opponent.update_root(next_position, copy.deepcopy(new_game.get_board()), copy.deepcopy(not new_game.player))

            # end of a game, print result:
            new_game.show_board()
            turn = 0
            self.score_board(i, new_game.result())
            del self.user
            del self.opponent

if __name__ == '__main__':
    runner = Runner(10, user_path='player_wan.User', opponent_path='player_random.User')
    runner.run_game(tree_visualization=False)