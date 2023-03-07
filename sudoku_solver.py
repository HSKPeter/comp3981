import copy
import math

NINE_X_NINE = [[0,0,3,0,2,0,6,0,0],[9,0,0,3,0,5,0,0,1],[0,0,1,8,0,6,4,0,0],[0,0,8,1,0,2,9,0,0],[7,0,0,0,0,0,0,0,8],[0,0,6,7,0,8,2,0,0],[0,0,2,6,0,9,5,0,0],[8,0,0,2,0,3,0,0,9],[0,0,5,0,1,0,3,0,0]]
TWELVE_X_TWELVE = [[0,0,0,11,8,0,0,0,0,0,0,6],[10,0,0,0,0,0,0,0,0,4,0,11],[0,7,0,0,0,0,2,6,0,0,1,0],[7,0,12,0,0,0,9,0,0,0,0,0],[0,0,0,10,6,2,0,0,5,0,0,0],[0,9,0,8,0,10,0,0,0,0,0,0],[0,1,0,0,5,0,7,8,0,0,6,3],[0,0,0,7,2,0,10,0,8,0,12,5],[0,6,0,5,0,0,12,0,0,1,0,0],[6,5,7,12,0,3,8,2,0,11,0,0],[0,0,8,0,0,1,5,0,0,7,3,9],[9,3,4,1,0,0,6,10,12,0,0,0]]
SIXTEEN_X_SIXTEEN = [[0,3,0,0,0,0,0,2,0,0,10,14,13,0,0,0],[0,0,0,0,0,0,0,0,16,15,0,1,0,0,0,6],[0,11,0,10,9,1,0,0,0,0,0,2,0,14,0,0],[0,0,0,8,0,16,0,14,0,0,0,0,0,0,9,0],[0,0,0,0,15,0,13,7,0,0,12,0,0,0,0,10],[0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,14,0,0,0,4,0,13,0,0,0,12],[0,13,0,0,0,0,0,0,0,0,5,0,0,0,0,0],[0,0,0,6,0,0,9,0,14,0,0,0,0,13,1,0],[15,0,0,0,5,0,3,6,0,0,0,0,0,2,0,0],[13,16,2,0,0,0,0,0,3,12,6,0,0,0,0,9],[0,0,0,0,0,13,0,4,9,0,1,7,0,8,6,0],[16,15,0,0,0,0,0,0,8,1,0,10,0,0,7,2],[0,0,0,11,0,0,0,0,2,14,0,0,0,0,0,4],[5,10,0,2,16,12,7,0,15,11,4,6,0,9,0,14],[1,0,0,4,2,8,0,9,0,5,16,3,11,12,10,0]]
TWENTY_FIVE_X_TWENTY_FIVE = [[0,0,0,0,0,20,0,0,9,0,25,14,0,0,0,0,0,7,0,0,0,0,0,0,0],[21,0,6,18,20,1,0,0,0,0,0,11,4,0,0,0,0,0,0,0,0,14,0,0,22],[0,0,0,3,22,0,8,0,0,7,23,0,0,2,0,0,0,0,0,0,0,0,0,0,0],[0,10,0,0,0,0,3,6,0,0,0,0,0,0,0,0,0,0,18,0,0,24,11,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,21,0,0,9,10,0,0,0],[0,0,0,0,0,0,24,0,0,14,0,0,12,0,17,0,1,0,6,25,0,0,0,21,0],[0,0,14,0,24,12,0,0,0,0,0,0,0,0,6,0,0,2,0,22,10,0,7,3,0],[0,0,0,0,0,22,0,0,0,0,14,21,0,25,0,0,0,0,0,0,0,0,0,0,0],[6,21,18,0,0,0,0,0,0,0,11,0,10,0,0,0,16,0,5,0,0,0,12,0,0],[0,0,0,0,10,0,0,0,0,0,0,0,0,0,9,0,0,0,0,0,0,5,2,0,0],[0,0,0,0,0,0,2,17,0,0,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,25,0,0,0,4,0,16,3,0,9,0,22,0,18,0,17,15,14,0,11,0,24,0],[8,0,0,0,14,0,25,19,23,0,0,0,15,0,0,0,21,0,0,0,4,3,9,10,7],[0,4,0,0,0,24,0,7,5,0,0,17,0,0,2,1,6,0,0,20,0,21,19,16,13],[0,13,0,0,2,0,0,0,0,0,0,0,25,0,0,0,7,0,0,4,1,0,0,18,6],[11,0,22,9,4,0,5,0,0,0,0,12,0,17,23,25,0,0,0,6,13,19,16,15,0],[24,0,0,17,0,0,0,0,0,18,0,1,0,0,0,22,2,0,13,21,7,9,10,0,0],[0,2,15,5,0,11,0,0,22,0,21,16,0,3,0,0,4,12,0,0,0,0,18,0,0],[0,18,20,0,0,17,1,2,0,0,0,0,11,0,0,0,9,16,3,0,0,25,0,0,8],[10,12,7,25,0,0,16,0,3,6,0,13,0,9,0,15,0,0,17,0,0,0,24,5,0],[3,0,11,4,9,0,0,8,0,5,0,0,0,12,0,6,0,25,20,0,0,0,15,19,0],[14,8,24,16,17,0,19,12,0,23,6,0,0,0,5,21,0,22,0,11,0,7,0,0,10],[2,0,19,13,5,0,0,0,0,20,0,0,21,16,0,7,0,0,0,12,0,6,0,0,0],[0,20,23,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,9,16,0,12,22,25,14],[12,0,0,0,0,0,21,3,0,0,9,0,0,13,22,0,17,15,14,18,0,0,0,0,0]]

RAW_BOARD = NINE_X_NINE

SQUARE_ROOTS = {
    9: 3,
    16: 4,
    25: 5,
    100: 10
}


class SudokuSolver:
    def __init__(self, raw_board: [[int]]) -> None:
        self.board = raw_board
        self.stack = [Node(self.board)]

    def solve(self):
        i = 0
        stack_size = len(self.stack)
        while stack_size > 0:
            current_node = self.stack[-1]
            if i % 1000 == 0:
                print(current_node)
            if current_node.is_solution():
                print(f"result found: i = {i}")
                return current_node
            next_node = current_node.get_first_unchecked_child()
            if next_node is None:
                self.stack.pop()
                current_node.check()
            else:
                self.stack.append(next_node)
            i += 1
            stack_size = len(self.stack)
            if i % 1000 == 0:
                print(f"searching (i = {i}; stack size = {stack_size})")


class NodeCollection:
    def __init__(self):
        self._list = list()

    def add(self, node):
        self._list.append(node)

    def __contains__(self, item):
        for node in self._list:
            if node.__str__() == item.__str__():
                return True

        return False


class Node:
    def __init__(self, board: [[int]], parent_node=None) -> None:
        self.board = board
        self.board_length = len(self.board)
        self.children = list()
        self._checked_children = NodeCollection()
        self.parent = parent_node

    def check(self):
        if self.parent is not None:
            self.parent.add_checked_child(self)

    def add_checked_child(self, invalid_child):
        self._checked_children.add(invalid_child)

    def __str__(self):
        result = ""
        for row in self.board:
            for cell in row:
                cell_representation = cell if cell != 0 else "__"
                new_part = f"{cell_representation} "
                result += new_part.ljust(4)
            result += "\n"
        return result

    def is_solution(self):
        """
        To become a solution, it is required that:
        (1) All cells are filled
        (2) No duplicated value in each row
        (3) No duplicated value in each column
        (4) No duplicated value in each sub square
        """
        sqr_root = SQUARE_ROOTS[self.board_length]
        col_values_sets = [set() for i in range(self.board_length)]
        sub_square_values_sets = [set() for i in range(self.board_length)]
        for row_index in range(self.board_length):
            row_values_set = set()
            for col_index in range(self.board_length):
                sub_square_index = math.floor(row_index / sqr_root) * sqr_root + math.floor(col_index / sqr_root)
                sub_square_values_set = sub_square_values_sets[sub_square_index]
                cell = self.board[row_index][col_index]
                col_values_set = col_values_sets[col_index]
                if (cell == 0) or \
                   (cell in row_values_set) or \
                   (cell in col_values_set) or \
                   (cell in sub_square_values_set):
                    return False
        return True

    def get_first_unchecked_child(self):
        for node in self.generate_children():
            if node not in self._checked_children:
                return node

    def is_valid_insertion(self, row, col, new_value):
        rows = self.get_rows()
        if new_value in rows[row]:
            return False

        columns = self.get_columns()
        if new_value in columns[col]:
            return False

        sub_index = self.get_sub_square_index(row, col)
        squares = self.get_subsquares()
        if new_value in squares[sub_index]:
            return False

        return True

    def get_subsquares(self):
        board = self.board
        n = len(board)
        subsquares = []
        subsize = int(n ** 0.5)
        for i in range(0, n, subsize):
            for j in range(0, n, subsize):
                subsquare = set()
                for x in range(i, i + subsize):
                    for y in range(j, j + subsize):
                        subsquare.add(board[x][y])
                subsquares.append(subsquare)
        return subsquares

    def get_sub_square_index(self, row, col):
        subsize = int(math.sqrt(len(self.board)))
        subRow = row // subsize
        subCol = col // subsize
        subIndex = subRow * subsize + subCol
        return subIndex

    def get_rows(self):
        board = self.board
        rows = []
        for i in range(len(board)):
            row = set(board[i])
            rows.append(row)
        return rows

    def get_columns(self):
        board = self.board
        columns = []
        for j in range(len(board)):
            column = set()
            for i in range(len(board)):
                column.add(board[i][j])
            columns.append(column)
        return columns


def main():
    solver = SudokuSolver(RAW_BOARD)
    solution_node = solver.solve()
    print(f"solution:\n{solution_node}")
    if solution_node is not None:
        board = solution_node.board
        print(f"2D array: {board}")


if __name__ == '__main__':
    main()
