from mcts import MonteCarloTreeSearch, TreeNode
from net import Net
from const import *
from game import Board


class Play(object):
    def __init__(self):
        net = Net()
        if USECUDA:
            net = net.cuda()
        net.load_model("model.pt", cuda=USECUDA)
        self.net = net
        self.net.eval()

    def go(self):
        print("One rule:\r\n Move piece form 'x,y' \r\n eg 1,3\r\n")
        print("-" * 60)
        print("Ready Go")

        mc = MonteCarloTreeSearch(self.net, 1000)
        node = TreeNode()
        board = Board()

        while True:
            if board.c_player == BLACK:
                action = input(f"Your piece is 'O' and move: ")
                action = [int(n, 10) for n in action.split(",")]
                action = action[0] * board.size + action[1]
                next_node = TreeNode(action=action)
            else:
                _, next_node = mc.search(board, node)

            board.move(next_node.action)
            board.show()

            next_node.parent = None
            node = next_node

            if board.is_draw():
                print("-" * 28 + "Draw" + "-" * 28)
                return

            if board.is_game_over():
                if board.c_player == BLACK:
                    print("-" * 28 + "Win" + "-" * 28)
                else:
                    print("-" * 28 + "Loss" + "-" * 28)
                return

            board.trigger()


if __name__ == "__main__":
    p = Play()
    p.go()
