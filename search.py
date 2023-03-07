import bisect
import copy

SQUARE_ROOTS = {
    9: 3,
    16: 4,
    25: 5,
    100: 10
}

# class Foo:
#     heuristic_value_counter = 0

#     @classmethod
#     def foo(cls):
#         v = cls.heuristic_value_counter
#         cls.heuristic_value_counter += 1
#         return v

NINE = [[0,0,3,0,2,0,6,0,0],[9,0,0,3,0,5,0,0,1],[0,0,1,8,0,6,4,0,0],[0,0,8,1,0,2,9,0,0],[7,0,0,0,0,0,0,0,8],[0,0,6,7,0,8,2,0,0],[0,0,2,6,0,9,5,0,0],[8,0,0,2,0,3,0,0,9],[0,0,5,0,1,0,3,0,0]]
SIXTEEN = [[0,3,0,0,0,0,0,2,0,0,10,14,13,0,0,0],[0,0,0,0,0,0,0,0,16,15,0,1,0,0,0,6],[0,11,0,10,9,1,0,0,0,0,0,2,0,14,0,0],[0,0,0,8,0,16,0,14,0,0,0,0,0,0,9,0],[0,0,0,0,15,0,13,7,0,0,12,0,0,0,0,10],[0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,14,0,0,0,4,0,13,0,0,0,12],[0,13,0,0,0,0,0,0,0,0,5,0,0,0,0,0],[0,0,0,6,0,0,9,0,14,0,0,0,0,13,1,0],[15,0,0,0,5,0,3,6,0,0,0,0,0,2,0,0],[13,16,2,0,0,0,0,0,3,12,6,0,0,0,0,9],[0,0,0,0,0,13,0,4,9,0,1,7,0,8,6,0],[16,15,0,0,0,0,0,0,8,1,0,10,0,0,7,2],[0,0,0,11,0,0,0,0,2,14,0,0,0,0,0,4],[5,10,0,2,16,12,7,0,15,11,4,6,0,9,0,14],[1,0,0,4,2,8,0,9,0,5,16,3,11,12,10,0]]
RAW_BOARD = NINE

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

    def __len__(self):
        return len(self.frontier)


class Node:
    def __init__(self, board: [[int]], action=(None, None, None), parent_node=None) -> None:
        self.board = board
        self.action = action
        self.board_length = len(self.board)
        self.state = self.compute_state()
        self.row_values_sets = [set() for _ in range(self.board_length)]
        self.col_values_sets = [set() for _ in range(self.board_length)]
        self.sub_square_values_sets = [set() for _ in range(self.board_length)]

        self.empty_cell_count = 0

        for row in range(self.board_length):
            for col in range(self.board_length):
                value = self.board[row][col]
                if value != 0:
                    sub_square_index = self.get_sub_square_index(row, col)
                    self.row_values_sets[row].add(value)
                    self.col_values_sets[col].add(value)
                    self.sub_square_values_sets[sub_square_index].add(value)
                else:
                    self.empty_cell_count += 1

        self.heuristic_value = self.compute_heuristics_value()

    def compute_state(self):
        result = ""
        for row in self.board:
            for cell in row:
                cell_representation = cell if cell != 0 else "__"
                new_part = f"{cell_representation} "
                result += new_part.ljust(4)
            result += "\n"
        return result

    def compute_heuristics_value(self):
        # return Foo.foo()

        h = 0
        # # for row in self.board:
        # #     for cell in row:
        # #         if cell == 0:
        # #             h += 1
        # # return -h

        for row in range(self.board_length):
            for col in range(self.board_length):
                value = self.board[row][col]
                sub_square_index = self.get_sub_square_index(row, col)
                if value != 0:
                    set_of_prohibited_values = self.row_values_sets[row].copy()
                    set_of_prohibited_values.update(self.col_values_sets[col])
                    set_of_prohibited_values.update(self.sub_square_values_sets[sub_square_index])
                    possible_values_count = self.board_length - len(set_of_prohibited_values)
                    h += possible_values_count
        return -h

    def is_solution(self):
        """
        To become a solution, it is required that:
        (1) All cells are filled
        (2) No duplicated value in each row
        (3) No duplicated value in each column
        (4) No duplicated value in each sub square
        """
        sqr_root = SQUARE_ROOTS[self.board_length]
        col_values_sets = [set() for _ in range(self.board_length)]
        sub_square_values_sets = [set() for _ in range(self.board_length)]
        for row_index in range(self.board_length):
            row_values_set = set()
            for col_index in range(self.board_length):
                sub_square_index = (row_index // sqr_root) * sqr_root + (col_index // sqr_root)
                sub_square_values_set = sub_square_values_sets[sub_square_index]
                cell = self.board[row_index][col_index]
                col_values_set = col_values_sets[col_index]
                if (cell == 0) or \
                   (cell in row_values_set) or \
                   (cell in col_values_set) or \
                   (cell in sub_square_values_set):
                    return False
                row_values_set.add(cell)
                col_values_set.add(cell)
                sub_square_values_set.add(cell)
        return True

    def expand(self):
        n = len(self.board)
        for row in range(n):
            for col in range(n):
                if self.board[row][col] == 0:
                    for i in range(1, n + 1):
                        if self.is_valid_insertion(row, col, i):
                            new_board = copy.deepcopy(self.board)
                            new_board[row][col] = i
                            yield Node(new_board, (row, col, i), self)

    def is_valid_insertion(self, row, col, value):
        sub_square_index = self.get_sub_square_index(row, col)
        if value in self.row_values_sets[row] or value in self.col_values_sets[col] or value in self.sub_square_values_sets[sub_square_index]:
            return False

        return True

    def get_sub_square_index(self, row, col):
        sub_size = SQUARE_ROOTS[self.board_length]
        sub_row = row // sub_size
        sub_col = col // sub_size
        sub_index = sub_row * sub_size + sub_col
        return sub_index

    # https://stackoverflow.com/a/26840843
    def __lt__(self, cmp_node):
        return self.heuristic_value < cmp_node.heuristic_value

    def __str__(self):
        row, col, value = self.action
        return f"Fill {value} at {row}, {col} (Empty cell count = {self.empty_cell_count})\n{self.state}"


class SudokuSolver:
    def __init__(self, raw_board: [[int]]) -> None:
        self.board = raw_board
        self.frontier = NodeFrontier(Node(self.board))
        self.reached_state = list()

    def solve(self):
        while self.frontier.is_not_empty() > 0:
            current_node = self.frontier.pop()
            print(f"Frontier size: {len(self.frontier)}")
            print(current_node)

            if current_node.is_solution():
                return current_node

            for child_node in current_node.expand():
                state = child_node.state
                if state not in self.reached_state:
                    self.reached_state.append(state)
                    self.frontier.add(child_node)

def main():
    solver = SudokuSolver(RAW_BOARD)
    solution_node = solver.solve()
    print(f"solution:\n{solution_node}")
    if solution_node is not None:
        board = solution_node.board
        print(f"2D array: {board}")


if __name__ == '__main__':
    main()