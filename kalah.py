import copy


def reverse_board(board):
    return board[7:] + board[:7]

class Kalah:
    def __init__(self, board=[4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0]):
        '''
         U  U  U  U  U  U US  O  O  O  O  O  O OS
        [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0]
        U: User side
        O: Opponent side
        US: User score well
        OS: Opponent score well
        '''
        self.board = board
        self.player = True
        self.game_over = False
        
    def get_board(self):
        if self.player:
            return self.board
        else:
            return reverse_board(self.board)
    
    def show_board(self, score=None, free_turn=False):
        print("\t\t<--- North")
        print("--------------------------------------------")
        print("  "+"\t".join([str(self.board[12-idx]) for idx in range(0, 6)]))
        print()
        print("  "+"\t\t\t\t\t".join([str(self.board[-1]), str(self.board[6])]))
        print()
        print("  "+"\t".join([str(self.board[idx]) for idx in range(0, 6)]))
        print("--------------------------------------------")
        print("\t\tSouth --->")
        if score is None:
            print("\n")
            return
        
        if score == 0:
            if self.player:
                p = "User"
            else:
                p = "Opponent"
            print("No score. Turn passed to "+p+".\n\n")
        elif score > 0:
            if free_turn:
                if self.player:
                    p = "User"
                else:
                    p = "Opponent"
                print("free turn!")
            else:
                if not self.player:
                    p = "User"
                else:
                    p = "Opponent"

            print(p+" got "+str(score)+" point(s).")
        
    def empty_taking(self, positions):
        if self.player:
            score_idx = 6
        else:
            score_idx = len(self.board)-1
        for i in positions:
            self.board[score_idx] += self.board[i]
            self.board[score_idx] += self.board[-2-i]
            self.board[i] = 0
            self.board[-2-i] = 0      
    
    def move(self, position):
        assert position >= 0 and position < 6
        if self.player:
            score_idx = 6
        else:
            position += 7
            score_idx = -1
        move_cnt = copy.deepcopy(self.board[position])
        assert move_cnt > 0
        current_score = copy.deepcopy(self.board[score_idx])

        take_opponents = []
        pos = position
        last_position = pos
        for i in range(1, move_cnt+1):
            if self.player and pos + i == len(self.board)-1:
                pos = -i
            elif not self.player and pos + i == len(self.board):
                pos = -i
            elif not self.player and pos + i == 6:
                pos += 1
            self.board[pos+i] += 1
            last_position = pos + i
        self.board[position] -= move_cnt
        if self.player and last_position < 6 and self.board[last_position] == 1:
            take_opponents.append(last_position)
        elif not self.player and 5 < last_position and last_position < len(self.board)-1 and self.board[last_position] == 1:
            take_opponents.append(last_position)
        
        self.empty_taking(take_opponents)
        
        if (self.player and last_position == 6) or (not self.player and last_position == len(self.board)-1):
            free_turn = True
            pass
        else:
            self.player = not self.player
            free_turn = False
                    
        return self.board[score_idx]-current_score, free_turn

    def is_game_over(self):
        if sum(self.board[:6]) == 0 or sum(self.board[7:-1]) == 0:
            self.game_over = True
        return self.game_over
        
    def result(self):
        north_score = sum(self.board[7:])
        south_score = sum(self.board[:7])
        if self.game_over:
            if north_score < south_score:
                return 1
            elif north_score == south_score:
                return 0
            else:
                return -1
            
        return None

