import bisect
class NodeFrontier:
    def __init__(self, root_node):
        self.frontier = list()
        self.frontier.append(root_node)

    def is_not_empty(self):
        return len(self.frontier) != 0

    def pop(self):
        return self.frontier.pop()

    def add(self, node):
        bisect.insort_left(self.frontier, node)


class Node:
    def __init__(self, board: [[int]], parent_node=None) -> None:
        self.board = board
        self.board_length = len(self.board)
        self.state = self.compute_state()
        self.heuristic_value = self.compute_heuristics_value()

    def compute_state(self):
        pass

    def compute_heuristics_value(self):
        pass

    def is_solution(self):
        pass

    def expand(self):
        pass

    # https://stackoverflow.com/a/26840843
    def __lt__(self, cmp_node):
        return self.heuristic_value < cmp_node.heuristic_value

    def __str__(self):
        return self.state


class SudokuSolver:
    def __init__(self, raw_board: [[int]]) -> None:
        self.board = raw_board
        self.frontier = NodeFrontier(Node(self.board))
        self.reached_state = list()

    def solve(self):
        while self.frontier.is_not_empty() > 0:
            current_node = self.frontier.pop()
            print(current_node)

            if current_node.is_solution():
                return current_node

            for child_node in current_node.expand():
                state = child_node.state
                if state not in self.reached_state:
                    self.reached_state.append(state)
                    self.frontier.add(child_node)

def main():
    pass


if __name__ == '__main__':
    main()