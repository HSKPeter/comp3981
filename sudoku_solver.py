import copy
import math

RAW_BOARD = [
    [0, 0, 1, 2],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 1, 0, 0]
]


class SudokuSolver:
    def __init__(self, raw_board: [[int]]) -> None:
        self.board = raw_board
        self.stack = [Node(self.board)]

    def solve(self):
        while len(self.stack) > 0:
            current_node = self.stack[-1]
            if current_node.is_solution():
                return current_node.board
            next_node = current_node.get_first_unchecked_child()
            if next_node is None:
                self.stack.pop()
                current_node.check()
            else:
                self.stack.append(next_node)


class Node:
    def __init__(self, board: [[int]]) -> None:
        self.board = board
        self.children = []
        self.find_valid_children()

        self.isChecked = False

    def check(self):
        self.isChecked = True

    def is_solution(self):
        for row in self.board:
            for cell in row:
                if cell == 0:
                    return False
        return True

    def get_first_unchecked_child(self):
        for node in self.children:
            if not node.isChecked:
                return node
        return None

    def find_valid_children(self):
        n = len(self.board)
        for row in range(n):
            for col in range(n):
                if self.board[row][col] == 0:
                    for i in range(1, n + 1):
                        if self.is_valid_insertion(row, col, i):
                            new_board = copy.deepcopy(self.board)
                            new_board[row][col] = i
                            self.children.append(Node(new_board))
                    return

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
    solution = solver.solve()
    print("solution: ", solution)


if __name__ == '__main__':
    main()
