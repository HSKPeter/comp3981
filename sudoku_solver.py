import copy
from typing import List
import random
from datetime import datetime, timedelta
import time

NINE_X_NINE = [[0, 0, 3, 0, 2, 0, 6, 0, 0], [9, 0, 0, 3, 0, 5, 0, 0, 1], [0, 0, 1, 8, 0, 6, 4, 0, 0],
               [0, 0, 8, 1, 0, 2, 9, 0, 0], [
                   7, 0, 0, 0, 0, 0, 0, 0, 8], [0, 0, 6, 7, 0, 8, 2, 0, 0], [0, 0, 2, 6, 0, 9, 5, 0, 0],
               [8, 0, 0, 2, 0, 3, 0, 0, 9], [0, 0, 5, 0, 1, 0, 3, 0, 0]]
TWELVE_X_TWELVE = [[10, 11, 8, 12, 5, 6, 3, 4, 1, 7, 2, 9], [9, 4, 2, 5, 1, 11, 7, 10, 6, 12, 3, 8], [1, 3, 6, 7, 8, 12, 2, 9, 10, 5, 4, 11], [2, 6, 12, 4, 3, 9, 8, 5, 7, 11, 1, 10], [5, 1, 11, 3, 7, 2, 10, 12, 8, 9, 6, 4], [8, 10, 7, 9, 6, 4, 11, 1, 12, 3, 5, 2], [4, 9, 5, 11, 10, 7, 1, 2, 3, 6, 8, 12], [6, 8, 3, 1, 4, 5, 12, 11, 9, 2, 10, 7], [7, 12, 10, 2, 9, 3, 6, 8, 4, 1, 11, 5], [11, 7, 9, 8, 2, 1, 4, 3, 5, 10, 12, 6], [3, 2, 4, 6, 12, 10, 5, 7, 11, 8, 9, 1], [12, 5, 1, 10, 11, 8, 9, 6, 2, 4, 7, 3]]
SIXTEEN_X_SIXTEEN_SOLVED = [[15, 8, 16, 14, 9, 13, 6, 1, 5, 3, 7, 4, 10, 2, 11, 12],
                            [6, 12, 11, 7, 5, 3, 10, 8, 1, 14, 9, 2, 13, 4, 16, 15],
                            [5, 13, 3, 10, 7, 16, 2, 4, 8, 11, 15, 12, 9, 6, 1, 14],
                            [9, 1, 2, 4, 11, 15, 14, 12, 16, 10, 13, 6, 8, 5, 7, 3],
                            [8, 7, 14, 13, 16, 5, 11, 10, 12, 15, 4, 1, 2, 3, 6, 9],
                            [16, 3, 10, 11, 13, 14, 4, 15, 7, 2, 6, 9, 5, 1, 12, 8],
                            [1, 9, 5, 15, 8, 6, 12, 2, 10, 13, 3, 11, 14, 16, 4, 7],
                            [2, 6, 4, 12, 3, 7, 1, 9, 14, 16, 5, 8, 15, 11, 13, 10], [
                                10, 4, 8, 16, 14, 12, 5, 7, 9, 1, 11, 13, 6, 15, 3, 2],
                            [11, 2, 9, 5, 15, 4, 16, 13, 6, 12, 8, 3, 7, 14, 10, 1],
                            [12, 15, 7, 6, 1, 10, 8, 3, 2, 4, 14, 5, 16, 13, 9, 11],
                            [13, 14, 1, 3, 2, 11, 9, 6, 15, 7, 16, 10, 12, 8, 5, 4],
                            [14, 5, 12, 1, 6, 8, 13, 11, 3, 9, 2, 7, 4, 10, 15, 16],
                            [3, 10, 6, 9, 4, 1, 15, 14, 13, 8, 12, 16, 11, 7, 2, 5],
                            [7, 11, 13, 8, 12, 2, 3, 16, 4, 5, 10, 15, 1, 9, 14, 6],
                            [4, 16, 15, 2, 10, 9, 7, 5, 11, 6, 1, 14, 3, 12, 8, 13]]
# TWENTY_FIVE_X_TWENTY_FIVE = [[0, 0, 0, 0, 0, 20, 0, 0, 9, 0, 25, 14, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0],
#                              [21, 0, 6, 18, 20, 1, 0, 0, 0, 0, 0, 11, 4, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 22],
#                              [0, 0, 0, 3, 22, 0, 8, 0, 0, 7, 23, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                              [0, 10, 0, 0, 0, 0, 3, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 24, 11, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 9, 10, 0, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 24, 0, 0, 14, 0, 0, 12, 0, 17, 0, 1, 0, 6, 25, 0, 0, 0, 21, 0],
#                              [0, 0, 14, 0, 24, 12, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 2, 0, 22, 10, 0, 7, 3, 0],
#                              [0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 14, 21, 0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                              [6, 21, 18, 0, 0, 0, 0, 0, 0, 0, 11, 0, 10, 0, 0, 0, 16, 0, 5, 0, 0, 0, 12, 0, 0],
#                              [0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 5, 2, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 2, 17, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                              [0, 0, 25, 0, 0, 0, 4, 0, 16, 3, 0, 9, 0, 22, 0, 18, 0, 17, 15, 14, 0, 11, 0, 24, 0],
#                              [8, 0, 0, 0, 14, 0, 25, 19, 23, 0, 0, 0, 15,
#                               0, 0, 0, 21, 0, 0, 0, 4, 3, 9, 10, 7],
#                              [0, 4, 0, 0, 0, 24, 0, 7, 5, 0, 0, 17, 0, 0, 2, 1, 6, 0, 0, 20, 0, 21, 19, 16, 13],
#                              [0, 13, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 7, 0, 0, 4, 1, 0, 0, 18, 6],
#                              [11, 0, 22, 9, 4, 0, 5, 0, 0, 0, 0, 12, 0, 17, 23, 25, 0, 0, 0, 6, 13, 19, 16, 15, 0],
#                              [24, 0, 0, 17, 0, 0, 0, 0, 0, 18, 0, 1, 0, 0, 0, 22, 2, 0, 13, 21, 7, 9, 10, 0, 0],
#                              [0, 2, 15, 5, 0, 11, 0, 0, 22, 0, 21, 16, 0, 3, 0, 0, 4, 12, 0, 0, 0, 0, 18, 0, 0],
#                              [0, 18, 20, 0, 0, 17, 1, 2, 0, 0, 0, 0, 11, 0, 0, 0, 9, 16, 3, 0, 0, 25, 0, 0, 8],
#                              [10, 12, 7, 25, 0, 0, 16, 0, 3, 6, 0, 13, 0, 9, 0, 15, 0, 0, 17, 0, 0, 0, 24, 5, 0],
#                              [3, 0, 11, 4, 9, 0, 0, 8, 0, 5, 0, 0, 0, 12, 0, 6, 0, 25, 20, 0, 0, 0, 15, 19, 0],
#                              [14, 8, 24, 16, 17, 0, 19, 12, 0, 23, 6, 0, 0, 0, 5, 21, 0, 22, 0, 11, 0, 7, 0, 0, 10],
#                              [2, 0, 19, 13, 5, 0, 0, 0, 0, 20, 0, 0, 21, 16, 0, 7, 0, 0, 0, 12, 0, 6, 0, 0, 0],
#                              [0, 20, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 9, 16, 0, 12, 22, 25, 14],
#                              [12, 0, 0, 0, 0, 0, 21, 3, 0, 0, 9, 0, 0, 13, 22, 0, 17, 15, 14, 18, 0, 0, 0, 0, 0]]

TWENTY_FIVE_X_TWENTY_FIVE = [[18, 14, 7, 6, 23, 4, 11, 3, 17, 13, 8, 21, 12, 5, 16, 22, 19, 10, 24, 1, 15, 25, 2, 9, 20], [1, 19, 22, 24, 3, 5, 21, 10, 16, 7, 23, 14, 25, 9, 6, 15, 2, 8, 4, 20, 17, 11, 13, 12, 18], [10, 11, 15, 5, 21, 20, 8, 14, 24, 9, 18, 1, 22, 17, 2, 23, 25, 6, 12, 13, 19, 7, 4, 16, 3], [20, 25, 16, 9, 13, 23, 2, 12, 22, 19, 15, 10, 24, 4, 7, 3, 11, 17, 14, 18, 21, 6, 5, 1, 8], [12, 17, 8, 4, 2, 25, 15, 18, 1, 6, 11, 13, 20, 3, 19, 5, 21, 16, 9, 7, 24, 10, 23, 14, 22], [22, 1, 24, 3, 19, 7, 16, 5, 10, 21, 9, 6, 14, 25, 23, 8, 4, 20, 15, 2, 18, 13, 17, 11, 12], [7, 18, 6, 23, 14, 13, 17, 4, 3, 11, 5, 16, 21, 12, 8, 10, 24, 1, 22, 19, 20, 2, 15, 25, 9], [8, 12, 4, 2, 17, 6, 1, 25, 18, 15, 3, 19, 13, 20, 11, 16, 9, 7, 5, 21, 22, 23, 24, 10, 14], [15, 10, 5, 21, 11, 9, 24, 20, 14, 8, 17, 2, 1, 22, 18, 6, 12, 13, 23, 25, 3, 4, 19, 7, 16], [16, 20, 9, 13, 25, 19, 22, 23, 12, 2, 4, 7, 10, 24, 15, 17, 14, 18, 3, 11, 8, 5, 21, 6, 1], [9, 13, 25, 16, 20, 2, 23, 22, 19, 12, 7, 15, 4, 10, 24, 18, 3, 14, 11, 17, 5, 21, 1, 8, 6], [4, 2, 17, 8, 12, 15, 25, 1, 6, 18, 19, 11, 3, 13, 20, 7, 5, 9, 21, 16, 23, 24, 14, 22, 10], [24, 3, 19, 22, 1, 21, 5, 16, 7, 10, 6, 23, 9, 14, 25, 20, 15, 4, 2, 8, 13, 17, 12, 18, 11], [6, 23, 14, 7, 18, 11, 4, 17, 13, 3, 16, 8, 5, 21, 12, 1, 22, 24, 19, 10, 2, 15, 9, 20, 25], [5, 21, 11, 15, 10, 8, 20, 24, 9, 14, 2, 18, 17, 1, 22, 13, 23, 12, 25, 6, 4, 19, 16, 3, 7], [11, 5, 21, 10, 15, 24, 14, 9, 8, 20, 22, 17, 2, 18, 1, 12, 13, 25, 6, 23, 7, 16, 3, 19, 4], [25, 9, 13, 20, 16, 22, 12, 19, 2, 23, 24, 4, 7, 15, 10, 14, 18, 11, 17, 3, 6, 1, 8, 21, 5], [14, 6, 23, 18, 7, 17, 3, 13, 11, 4, 12, 5, 16, 8, 21, 24, 1, 19, 10, 22, 25, 9, 20, 15, 2], [17, 4, 2, 12, 8, 1, 18, 6, 15, 25, 20, 3, 19, 11, 13, 9, 7, 21, 16, 5, 10, 14, 22, 24, 23], [19, 24, 3, 1, 22, 16, 10, 7, 21, 5, 25, 9, 6, 23, 14, 4, 20, 2, 8, 15, 11, 12, 18, 17, 13], [21, 15, 10, 11, 5, 14, 9, 8, 20, 24, 1, 22, 18, 2, 17, 25, 6, 23, 13, 12, 16, 3, 7, 4, 19], [13, 16, 20, 25, 9, 12, 19, 2, 23, 22, 10, 24, 15, 7, 4, 11, 17, 3, 18, 14, 1, 8, 6, 5, 21], [23, 7, 18, 14, 6, 3, 13, 11, 4, 17, 21, 12, 8, 16, 5, 19, 10, 22, 1, 24, 9, 20, 25, 2, 15], [2, 8, 12, 17, 4, 18, 6, 15, 25, 1, 13, 20, 11, 19, 3, 21, 16, 5, 7, 9, 14, 22, 10, 23, 24], [3, 22, 1, 19, 24, 10, 7, 21, 5, 16, 14, 25, 23, 6, 9, 2, 8, 15, 20, 4, 12, 18, 11, 13, 17]]

FLOOR_SQUARE_ROOTS = {
    9: 3,
    12: 3,
    16: 4,
    25: 5,
    100: 10
}

class SolverExecutionExpiredException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class PuzzleUnsolvedException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class SudokuSolver:
    def __init__(self, raw_board: List[List[int]]) -> None:
        self.board = raw_board
        self.stack = [Node(self.board)]
        # TODO create new class the manages the insertion and sorting of nodes based on heuristics values
        self.reserved_stack = list()

    def solve(self, max_process_seconds=None):
        expiry_timestamp = (datetime.now() + timedelta(seconds=max_process_seconds)).timestamp() if max_process_seconds is not None else None

        i = 0
        timeout = 0
        stack_size = len(self.stack)
        while stack_size > 0:
            if (expiry_timestamp is not None) and (time.time() >= expiry_timestamp):
                raise SolverExecutionExpiredException(f"No solution is found within {max_process_seconds} seconds")

            current_node = self.stack[-1]
            if current_node.is_solution():
                print(f"result found: i = {i}")
                return current_node
            current_node.expand()
            next_node = current_node.get_first_traversable_child()
            if next_node is None:
                current_node.check()
                self.stack.remove(current_node)
            else:
                self.stack.append(next_node)
            i += 1
            timeout += 1
            if timeout >= 5000:
                timeout = 0
                self.migrate_nodes_to_reserved_stack()

            if len(self.stack) == 0:
                self.migrate_nodes_in_reserved_stack()

            stack_size = len(self.stack)

            if i % 1000 == 0:
                print()
                print(f"searching (i = {i}; timeout = {timeout}; stack size = {stack_size}); reserved stack size = {len(self.reserved_stack)}")
                print(current_node)


    def migrate_nodes_to_reserved_stack(self):
        for node in self.stack[1:]:
            node.reserve()
            self.reserved_stack.append(node)

        self.stack = self.stack[:1]

    def migrate_nodes_in_reserved_stack(self):
        self.stack = [Node.mark_node_as_unreserved(node) for node in self.reserved_stack]
        self.reserved_stack = []


class Node:
    def __init__(self, board: List[List[int]]) -> None:
        self.board = board
        self.board_length = len(self.board)
        self.children = []
        self.domains = self.find_domains()
        self.heuristic_value = self.compute_heuristic_value()
        self.isChecked = False
        self.isExpanded = False
        self.is_reserved = False

    @staticmethod
    def mark_node_as_unreserved(node):
        node.unreserve()
        return node

    def compute_heuristic_value(self):
        # Sum the length of all domains
        return sum(len(domain) for row in self.domains for domain in row)

    def expand(self):
        if not self.isExpanded:
            self.find_valid_children()
            self.isExpanded = True

    def check(self):
        self.isChecked = True

    def reserve(self):
        self.is_reserved = True

    def unreserve(self):
        self.is_reserved = False

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
        for row in self.board:
            if 0 in row:
                return False
        return True

    # Traversable node means the node is not checked and not in reserved stack
    def get_first_traversable_child(self):
        for node in self.children:
            if not node.isChecked and (node.is_reserved is False):
                return node
        return None

    def find_valid_children(self):
        cell = self.find_min_domain_cell()
        if cell is None:
            return
        row, col = cell
        for i in self.domains[row][col]:
            if self.is_valid_insertion(row, col, i):
                new_board = copy.deepcopy(self.board)
                new_board[row][col] = i
                self.children.append(Node(new_board))
        self.children.sort()

    def find_min_domain_cell(self):
        board = self.board
        n = len(board)
        domains = self.domains
        min_domain_size = n + 1
        min_domain_cell = None

        board = self.board

        n = len(board)

        row_empty_values_counter = [0 for _ in range(self.board_length)]
        col_empty_values_counter = [0 for _ in range(self.board_length)]
        sub_empty_values_counter = [0 for _ in range(self.board_length)]
        for i in range(n):
            for j in range(n):
                if board[i][j] == 0:
                    k = self.get_sub_square_index(i, j)
                    row_empty_values_counter[i] += 1
                    col_empty_values_counter[j] += 1
                    sub_empty_values_counter[k] += 1

        empty_values_counters = (row_empty_values_counter, col_empty_values_counter, sub_empty_values_counter)

        for i in range(n):
            for j in range(n):
                if board[i][j] == 0 and len(domains[i][j]) < min_domain_size:
                    min_domain_size = len(domains[i][j])
                    min_domain_cell = (i, j)
                elif board[i][j] == 0 and len(domains[i][j]) == min_domain_size and min_domain_cell is not None:
                    min_domain_cell = self.find_cell_with_less_unassigned_neighbours(empty_values_counters, (i, j), min_domain_cell)
        return min_domain_cell

    def find_cell_with_less_unassigned_neighbours(self, empty_values_counters, cell1, cell2):
        row_index_1, col_index_1 = cell1
        row_index_2, col_index_2 = cell2

        sub_square_index_1 = self.get_sub_square_index(row_index_1, col_index_1)
        sub_square_index_2 = self.get_sub_square_index(row_index_2, col_index_2)

        row_empty_values_counter, col_empty_values_counter, sub_empty_values_counter = empty_values_counters

        cell1_empty_neighbours_count = row_empty_values_counter[row_index_1] + col_empty_values_counter[col_index_1] + sub_empty_values_counter[sub_square_index_1]
        cell2_empty_neighbours_count = row_empty_values_counter[row_index_2] + col_empty_values_counter[col_index_2] + sub_empty_values_counter[sub_square_index_2]

        if cell2_empty_neighbours_count <= cell1_empty_neighbours_count:
            return cell2

        return cell1

    def get_sub_square_index(self, row, col):
        board = self.board
        n = len(board)
        sub_n = FLOOR_SQUARE_ROOTS[n]  # size of each sub-square
        sub_m = n // sub_n  # number of sub-squares in each row or column
        sub_row = row // sub_m
        sub_col = col // sub_n
        sub_index = sub_row * sub_m + sub_col
        return sub_index

    def find_domains(self):
        board = self.board
        n = len(board)
        sub_n = FLOOR_SQUARE_ROOTS[n]  # size of each sub-square
        sub_m = n // sub_n  # number of sub-squares in each row or column
        domains = [[set(range(1, n + 1)) if board[i][j] == 0 else set()
                    for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(n):
                if board[i][j] == 0:
                    for k in range(n):
                        # remove values from the row and column
                        domains[i][j].discard(board[i][k])
                        domains[i][j].discard(board[k][j])
                    # remove values from the sub-square
                    sub_row = i // sub_n
                    sub_col = j // sub_m
                    for bi in range(sub_n):
                        for bj in range(sub_m):
                            i2 = sub_row * sub_n + bi
                            j2 = sub_col * sub_m + bj
                            domains[i][j].discard(board[i2][j2])
        return domains

    def is_valid_insertion(self, row, col, new_value):
        board = self.board
        n = len(board)
        sub_n = int(n ** 0.5)  # size of each sub-square
        sub_m = n // sub_n  # number of sub-squares in each row or column
        # check if the value i is already present in the same row or column
        for k in range(n):
            if board[row][k] == new_value or board[k][col] == new_value:
                return False
        # check if the value i is already present in the same sub-square
        sub_row = row // sub_n
        sub_col = col // sub_m
        for i in range(sub_n):
            for j in range(sub_m):
                if board[sub_row * sub_n + i][sub_col * sub_m + j] == new_value:
                    return False
        # if none of the checks failed, the insertion is valid
        return True

    def __gt__(self, cmp_node):
        if self.heuristic_value != cmp_node.heuristic_value:
            has_less_empty_cells = self.heuristic_value < cmp_node.heuristic_value
            return has_less_empty_cells


def change_values_to_zero(arr, p=0.75):
    """
    Takes an n x n array and changes p percent of the values to 0.

    Args:
        arr: list[list[int]] - an n x n array
        p: float - the proportion of values to change to 0 (default 0.75)

    Returns:
        list[list[int]] - the modified n x n array8
    """
    n = len(arr)
    m = int(p * n * n)  # number of values to change to 0
    indices = random.sample(range(n * n), m)  # choose random indices to change
    for idx in indices:
        i, j = divmod(idx, n)  # calculate the row and column indices
        arr[i][j] = 0  # set the value to 0
    return arr


def main():
    board = change_values_to_zero(TWENTY_FIVE_X_TWENTY_FIVE)
    solver = SudokuSolver(board)
    solution_node = solver.solve()
    print(f"puzzle:\n{board}")
    print(f"solution:\n{solution_node}")
    if solution_node:
        board = solution_node.board
        print(f"2D array: {board}")

def solve_with_brute_force(board):
    solver = SudokuSolver(board)
    solution_node = solver.solve(max_process_seconds=20)

    if solution_node is None:
        raise PuzzleUnsolvedException()

    return solution_node.board



if __name__ == '__main__':
    main()
