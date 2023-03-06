from types import Union
import copy

RAW_BOARD = [
    [0, 0, 1, 2],
    [2, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 1, 0, 0]
]


class SudokuSolver:
    def __init__(self, raw_board: [[int]]) -> None:
        self.board = self.initialize_board(raw_board)
        self.stack = [Node(self.board)]

    def initialize_board(self, raw_board: [[int]]):
        return [[Cell(value) for value in row] for row in raw_board]

    def solve(self):
        while len(node_stack) > 0:
            node = self.stack[-1]
            node.find_children()


# class Board:
#     def __init__(self, values: [[Cell]]):
#         self.values = values
#
#     def updateCell(self, cell, value):


class Cell:
    def __init__(self, value: int):
        self.value = value
        self.domain = list(range(1, 5))


class Node:
    def __init__(self, board: [[Cell]]) -> None:
        self.board = board
        # self.parent = parent
        self.children = []
        self.isChecked = False

    def check(self):
        self.isChecked = True

    def find_valid_children(self):
        next_cell = None
        for row in range(len(self.board)):
            for col in range(row):
                if self.board[row][col] == 0:
                    new_board = copy.deepcopy(self.board)
                    children = []
                    for i in range(1, 5):
                        new_board[row][col] = i
                        children.append(Node(new_board))
                        self.children = children
                        return self.children

    def is_valid_insertion(self, row, col, new_value):
        squares = self.getSubsquares()
        rows = self.getRows()
        columns = self.getColumns()
        if new_value in rows[row]:
            return False
        if new_value in columns[col]:
            return False
        

    def getSubsquares(self):
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

    def getRows(self):
        board = self.board
        rows = []
        for i in range(len(board)):
            row = set(board[i])
            rows.append(row)
        return rows

    def getColumns(self):
        board = self.board
        columns = []
        for j in range(len(board)):
            column = set()
            for i in range(len(board)):
                column.add(board[i][j])
            columns.append(column)
        return columns
