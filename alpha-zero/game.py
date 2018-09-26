from copy import deepcopy
import random

import numpy as np

from mcts import MonteCarloTreeSearch, TreeNode
from const import *


class Board(object):
    def __init__(self,
                 size=SIZE,
                 hist_num=HISTORY,
                 c_action=-1,
                 player=BLACK):

        self.size = size
        self.c_action = c_action
        self.hist_num = hist_num
        self.valid_moves = list(range(size**2))
        self.invalid_moves = []
        self.board = np.zeros([size, size])
        self.c_player = player
        self.players = {"black": BLACK, "white": WHITE}

        # BLACK -> 0 | WHITE -> 1
        self.history = [np.zeros((hist_num, size, size)),
                        np.zeros((hist_num, size, size))]

    # private method
    def _mask_pieces_by_player(self, player):
        '''binary feature planes'''

        new_board = np.zeros([self.size, self.size])
        new_board[np.where(self.board == player)] = 1.
        return new_board

    def _get_piece(self, x, y):
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.board[x, y]
        return SPACE

    def _is_space(self, x, y):
        assert 0 <= x < self.size and 0 <= y < self.size
        return self.board[x, y] == SPACE

    @property
    def last_player(self):
        if self.c_player == self.players["white"]:
            return self.players["black"]
        return self.players["white"]

    def clone(self):
        c_board = Board(size=self.size,
                        hist_num=self.hist_num,
                        player=self.c_player,
                        c_action=self.c_action)

        c_board.valid_moves = self.valid_moves.copy()
        c_board.invalid_moves = self.invalid_moves.copy()
        c_board.board = self.board.copy()
        c_board.history = [h.copy() for h in self.history]

        return c_board

    def move(self, action):
        x, y = action // self.size, action % self.size
        assert self._is_space(x, y)

        self.valid_moves.remove(action)
        self.invalid_moves.append(action)
        self.c_action = action
        self.board[x, y] = self.c_player

        p_index = int(self.c_player - BLACK)
        self.history[p_index] = np.roll(self.history[p_index], 1, axis=0)
        self.history[p_index][0] = self._mask_pieces_by_player(self.c_player)

    def is_game_over(self, player=None):
        x, y = self.c_action // self.size, self.c_action % self.size
        if player is None:
            player = self.c_player

        for i in range(x - 4, x + 5):
            if self._get_piece(i, y) == self._get_piece(i + 1, y) == self._get_piece(i + 2, y) == self._get_piece(i + 3, y) == self._get_piece(i + 4, y) == player:
                return True

        for j in range(y - 4, y + 5):
            if self._get_piece(x, j) == self._get_piece(x, j + 1) == self._get_piece(x, j + 2) == self._get_piece(x, j + 3) == self._get_piece(x, j + 4) == player:
                return True

        j = y - 4
        for i in range(x - 4, x + 5):
            if self._get_piece(i, j) == self._get_piece(i + 1, j + 1) == self._get_piece(i + 2, j + 2) == self._get_piece(i + 3, j + 3) == self._get_piece(i + 4, j + 4) == player:
                return True
            j += 1

        i = x + 4
        for j in range(y - 4, y + 5):
            if self._get_piece(i, j) == self._get_piece(i - 1, j + 1) == self._get_piece(i - 2, j + 2) == self._get_piece(i - 3, j + 3) == self._get_piece(i - 4, j + 4) == player:
                return True
            i -= 1

        return False

    def is_draw(self):
        index = np.where(self.board == SPACE)
        return len(index[0]) == 0

    def gen_state(self):
        to_action = np.zeros((1, self.size, self.size))
        to_action[0][self.c_action // self.size,
                     self.c_action % self.size] = 1.
        to_play = np.full((1, self.size, self.size), self.c_player - BLACK)
        state = np.concatenate(self.history + [to_play, to_action], axis=0)

        return state

    def trigger(self):
        self.c_player = self.players["black"] if self.c_player == self.players["white"] else self.players["white"]

    def show(self):
        for x in range(self.size):
            print("{0:8}".format(x), end='')
        print('\r\n')

        for row in range(self.size):
            print("{:4d}".format(row), end='')
            for col in range(self.size):
                if self.board[row, col] == SPACE:
                    print("-".center(8), end='')
                elif self.board[row, col] == BLACK:
                    print("O".center(8), end='')
                else:
                    print("X".center(8), end='')
            print('\r\n\r\n')


class Game(object):
    def __init__(self, net, evl_net):
        self.net = net
        self.evl_net = evl_net
        self.board = Board()

    def play(self):
        datas, node = [], TreeNode()
        mc = MonteCarloTreeSearch(self.net)
        move_count = 0

        while True:
            if move_count < TEMPTRIG:
                pi, next_node = mc.search(self.board, node, temperature=1)
            else:
                pi, next_node = mc.search(self.board, node)

            datas.append([self.board.gen_state(), pi, self.board.c_player])

            self.board.move(next_node.action)
            next_node.parent = None
            node = next_node

            if self.board.is_draw():
                reward = 0.
                break

            if self.board.is_game_over():
                reward = 1.
                break

            self.board.trigger()
            move_count += 1

        datas = np.asarray(datas)
        datas[:, 2][datas[:, 2] == self.board.c_player] = reward
        datas[:, 2][datas[:, 2] != self.board.c_player] = -reward

        return datas

    def evaluate(self, result):
        self.net.eval()
        self.evl_net.eval()

        if random.randint(0, 1) == 1:
            players = {
                BLACK: (MonteCarloTreeSearch(self.net), "net"),
                WHITE: (MonteCarloTreeSearch(self.evl_net), "eval"),
            }
        else:
            players = {
                WHITE: (MonteCarloTreeSearch(self.net), "net"),
                BLACK: (MonteCarloTreeSearch(self.evl_net), "eval"),
            }
        node = TreeNode()

        while True:
            _, next_node = players[self.board.c_player][0].search(
                self.board, node)

            self.board.move(next_node.action)

            if self.board.is_draw():
                result[0] += 1
                return

            if self.board.is_game_over():
                if players[self.board.c_player][1] == "net":
                    result[1] += 1
                else:
                    result[2] += 1
                return

            self.board.trigger()

            next_node.parent = None
            node = next_node

    def reset(self):
        self.board = Board()
