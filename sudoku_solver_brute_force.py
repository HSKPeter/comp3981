import copy
from typing import List, Set, Optional, Tuple
import random
from datetime import datetime, timedelta
import time
from algo_util import get_sub_square_index, FLOOR_SQUARE_ROOTS

NINE_X_NINE = [[0, 0, 3, 0, 2, 0, 6, 0, 0],
               [9, 0, 0, 3, 0, 5, 0, 0, 1],
               [0, 0, 1, 8, 0, 6, 4, 0, 0],
               [0, 0, 8, 1, 0, 2, 9, 0, 0],
               [7, 0, 0, 0, 0, 0, 0, 0, 8],
               [0, 0, 6, 7, 0, 8, 2, 0, 0],
               [0, 0, 2, 6, 0, 9, 5, 0, 0],
               [8, 0, 0, 2, 0, 3, 0, 0, 9],
               [0, 0, 5, 0, 1, 0, 3, 0, 0]]
TWELVE_X_TWELVE = [[10, 11, 8, 12, 5, 6, 3, 4, 1, 7, 2, 9], [9, 4, 2, 5, 1, 11, 7, 10, 6, 12, 3, 8],
                   [1, 3, 6, 7, 8, 12, 2, 9, 10, 5, 4, 11], [2, 6, 12, 4, 3, 9, 8, 5, 7, 11, 1, 10],
                   [5, 1, 11, 3, 7, 2, 10, 12, 8, 9, 6, 4], [8, 10, 7, 9, 6, 4, 11, 1, 12, 3, 5, 2],
                   [4, 9, 5, 11, 10, 7, 1, 2, 3, 6, 8, 12], [6, 8, 3, 1, 4, 5, 12, 11, 9, 2, 10, 7],
                   [7, 12, 10, 2, 9, 3, 6, 8, 4, 1, 11, 5], [11, 7, 9, 8, 2, 1, 4, 3, 5, 10, 12, 6],
                   [3, 2, 4, 6, 12, 10, 5, 7, 11, 8, 9, 1], [12, 5, 1, 10, 11, 8, 9, 6, 2, 4, 7, 3]]
SIXTEEN_X_SIXTEEN_SOLVED = [[15, 8, 16, 14, 9, 13, 6, 1, 5, 3, 7, 4, 10, 2, 11, 12],
                            [6, 12, 11, 7, 5, 3, 10, 8, 1, 14, 9, 2, 13, 4, 16, 15],
                            [5, 13, 3, 10, 7, 16, 2, 4, 8, 11, 15, 12, 9, 6, 1, 14],
                            [9, 1, 2, 4, 11, 15, 14, 12, 16, 10, 13, 6, 8, 5, 7, 3],
                            [8, 7, 14, 13, 16, 5, 11, 10, 12, 15, 4, 1, 2, 3, 6, 9],
                            [16, 3, 10, 11, 13, 14, 4, 15, 7, 2, 6, 9, 5, 1, 12, 8],
                            [1, 9, 5, 15, 8, 6, 12, 2, 10, 13, 3, 11, 14, 16, 4, 7],
                            [2, 6, 4, 12, 3, 7, 1, 9, 14, 16, 5, 8, 15, 11, 13, 10],
                            [10, 4, 8, 16, 14, 12, 5, 7, 9, 1, 11, 13, 6, 15, 3, 2],
                            [11, 2, 9, 5, 15, 4, 16, 13, 6, 12, 8, 3, 7, 14, 10, 1],
                            [12, 15, 7, 6, 1, 10, 8, 3, 2, 4, 14, 5, 16, 13, 9, 11],
                            [13, 14, 1, 3, 2, 11, 9, 6, 15, 7, 16, 10, 12, 8, 5, 4],
                            [14, 5, 12, 1, 6, 8, 13, 11, 3, 9, 2, 7, 4, 10, 15, 16],
                            [3, 10, 6, 9, 4, 1, 15, 14, 13, 8, 12, 16, 11, 7, 2, 5],
                            [7, 11, 13, 8, 12, 2, 3, 16, 4, 5, 10, 15, 1, 9, 14, 6],
                            [4, 16, 15, 2, 10, 9, 7, 5, 11, 6, 1, 14, 3, 12, 8, 13]]
TWENTY_FIVE_X_TWENTY_FIVE = [[0, 0, 0, 0, 0, 20, 0, 0, 9, 0, 25, 14, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0],
                             [21, 0, 6, 18, 20, 1, 0, 0, 0, 0, 0, 11, 4, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 22],
                             [0, 0, 0, 3, 22, 0, 8, 0, 0, 7, 23, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 10, 0, 0, 0, 0, 3, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 24, 11, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 9, 10, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 24, 0, 0, 14, 0, 0, 12, 0, 17, 0, 1, 0, 6, 25, 0, 0, 0, 21, 0],
                             [0, 0, 14, 0, 24, 12, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 2, 0, 22, 10, 0, 7, 3, 0],
                             [0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 14, 21, 0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [6, 21, 18, 0, 0, 0, 0, 0, 0, 0, 11, 0, 10, 0, 0, 0, 16, 0, 5, 0, 0, 0, 12, 0, 0],
                             [0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 5, 2, 0, 0],
                             [0, 0, 0, 0, 0, 0, 2, 17, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 25, 0, 0, 0, 4, 0, 16, 3, 0, 9, 0, 22, 0, 18, 0, 17, 15, 14, 0, 11, 0, 24, 0],
                             [8, 0, 0, 0, 14, 0, 25, 19, 23, 0, 0, 0, 15,
                              0, 0, 0, 21, 0, 0, 0, 4, 3, 9, 10, 7],
                             [0, 4, 0, 0, 0, 24, 0, 7, 5, 0, 0, 17, 0, 0, 2, 1, 6, 0, 0, 20, 0, 21, 19, 16, 13],
                             [0, 13, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 7, 0, 0, 4, 1, 0, 0, 18, 6],
                             [11, 0, 22, 9, 4, 0, 5, 0, 0, 0, 0, 12, 0, 17, 23, 25, 0, 0, 0, 6, 13, 19, 16, 15, 0],
                             [24, 0, 0, 17, 0, 0, 0, 0, 0, 18, 0, 1, 0, 0, 0, 22, 2, 0, 13, 21, 7, 9, 10, 0, 0],
                             [0, 2, 15, 5, 0, 11, 0, 0, 22, 0, 21, 16, 0, 3, 0, 0, 4, 12, 0, 0, 0, 0, 18, 0, 0],
                             [0, 18, 20, 0, 0, 17, 1, 2, 0, 0, 0, 0, 11, 0, 0, 0, 9, 16, 3, 0, 0, 25, 0, 0, 8],
                             [10, 12, 7, 25, 0, 0, 16, 0, 3, 6, 0, 13, 0, 9, 0, 15, 0, 0, 17, 0, 0, 0, 24, 5, 0],
                             [3, 0, 11, 4, 9, 0, 0, 8, 0, 5, 0, 0, 0, 12, 0, 6, 0, 25, 20, 0, 0, 0, 15, 19, 0],
                             [14, 8, 24, 16, 17, 0, 19, 12, 0, 23, 6, 0, 0, 0, 5, 21, 0, 22, 0, 11, 0, 7, 0, 0, 10],
                             [2, 0, 19, 13, 5, 0, 0, 0, 0, 20, 0, 0, 21, 16, 0, 7, 0, 0, 0, 12, 0, 6, 0, 0, 0],
                             [0, 20, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 9, 16, 0, 12, 22, 25, 14],
                             [12, 0, 0, 0, 0, 0, 21, 3, 0, 0, 9, 0, 0, 13, 22, 0, 17, 15, 14, 18, 0, 0, 0, 0, 0]]


class SolverExecutionExpiredException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class PuzzleUnsolvedException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class SudokuSolver:
    """
    Sudoku solver using brute force algorithm

    Args:
        board (List[List[int]]): unsolved sudoku board

    Attributes:
        board (List[List[int]]): sudoku board
        stack (List[Node]): stack of nodes representing the search tree
    """

    def __init__(self, board: List[List[int]]) -> None:
        self.board = board
        self.stack = [Node(self.board)]

    def solve(self, max_process_seconds=None, mute=True):
        """
        Solve the sudoku puzzle using brute force algorithm

        The algorithm iteratively searches for a solution by exploring different possibilities for filling the empty
        cells of the Sudoku board. Here's a brief explanation of how the function works:

        1. The algorithm uses a depth-first search approach, maintaining a stack of board states (nodes) as it explores
            different possibilities for filling the empty cells. The stack is initialized with the initial board state.

        2. The algorithm iterates until the stack is empty, which means it has explored all possibilities without
            finding a solution, or it finds a solution.

        3. Inside the loop, the algorithm checks if the current node (board state) is a solution. If it is, the function
            returns the solution.

        4. If the current node is not a solution, the algorithm expands the current node by generating child nodes
            representing the possibilities for filling the next empty cell.

        5. The algorithm then checks if there's any unexplored child node of the current node. If there is, it adds the
            child node to the stack and continues the search.

        6. If there's no unexplored child node, the current node is marked as checked, removed from the stack, and the
            search backtracks to the previous node.

        7. The algorithm also implements a timeout mechanism. If the stack doesn't change for 10000 iterations, it
            clears the stack (except for the initial node) and restarts the search from the beginning.

        8. The function also periodically prints the progress of the search, including the iteration count, timeout
            count, and stack size, if the mute parameter is set to False.

        Args:
            max_process_seconds: maximum number of seconds to run the algorithm
            mute: if True, the function will not print the progress of the search

        Returns:
            Node: solution node
        """
        expiry_timestamp = (datetime.now() + timedelta(seconds=max_process_seconds)).timestamp() \
            if max_process_seconds is not None else None

        i = 0
        timeout = 0
        stack_size = len(self.stack)
        while stack_size > 0:
            if (expiry_timestamp is not None) and (time.time() >= expiry_timestamp):
                raise SolverExecutionExpiredException(f"No solution is found within {max_process_seconds} seconds")

            current_node = self.stack[-1]
            if current_node.is_solution():
                if not mute:
                    print(f"result found: i = {i}")
                return current_node
            current_node.expand()
            next_node = current_node.get_first_unchecked_child()
            if next_node is None:
                current_node.check()
                self.stack.remove(current_node)
            else:
                self.stack.append(next_node)
            i += 1
            timeout += 1
            stack_size = len(self.stack)
            if timeout > 10000:
                timeout = 0
                for node in self.stack[1:]:
                    node.check()
                    self.stack.remove(node)
            if not mute and i % 1000 == 0:
                print()
                print(f"searching (i = {i}; timeout = {timeout}; stack size = {stack_size})")
                print(current_node)


class Node:
    """
    Node representing a board state

    Args:
        board (List[List[int]]): board state

    Attributes:
        board (List[List[int]]): board state
        board_length (int): length of the board
        children (List[Node]): list of child nodes
        domains (List[List[List[int]]]): list of domains for each cell
        heuristic_value (int): heuristic value of the node
        isChecked (bool): whether the node has been checked
        isExpanded (bool): whether the node has been expanded
    """

    def __init__(self, board: List[List[int]]) -> None:
        """
        Initialize the node

        Args:
            board: board state
        """
        self.board = board
        self.board_length = len(self.board)
        self.children = []
        self.domains = self.find_domains()
        self.heuristic_value = self.compute_heuristic_value()
        self.isChecked = False
        self.isExpanded = False

    def compute_heuristic_value(self) -> int:
        """
        Compute the heuristic value of the node by summing the length of each domain
        Returns:
            int: heuristic value
        """
        return sum(len(domain) for row in self.domains for domain in row)

    def expand(self):
        """
        Expand the node by generating child nodes representing the possibilities for filling the next empty cell
        """
        if not self.isExpanded:
            self.find_valid_children()
            self.isExpanded = True

    def check(self):
        """
        Mark the node as checked. Checked nodes will no longer be considered in the search.
        """
        self.isChecked = True

    def __str__(self) -> str:
        """
        String representation of the board state
        Returns:
            str: string representation of the board state
        """
        result = ""
        for row in self.board:
            for cell in row:
                cell_representation = cell if cell != 0 else "__"
                new_part = f"{cell_representation} "
                result += new_part.ljust(4)
            result += "\n"
        return result

    def is_solution(self) -> bool:
        """
        Check if the board state is a solution by checking if the board is filled.

        This method assumes that the board is valid.

        Returns:
            bool: True if the board state is a solution, False otherwise
        """
        for row in self.board:
            if 0 in row:
                return False
        return True

    def get_first_unchecked_child(self) -> Optional["Node"]:
        """
        Get the first unchecked child node

        Returns:
            Node: first unchecked child node
        """
        for node in self.children:
            if not node.isChecked:
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

    def find_min_domain_cell(self) -> Tuple[int, int]:
        """
        Find the cell with the smallest domain

        Returns:
            Tuple[int, int]: cell with the smallest domain
        """
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
                    min_domain_cell = self.find_cell_with_less_unassigned_neighbours(empty_values_counters, (i, j),
                                                                                     min_domain_cell)
        return min_domain_cell

    def find_cell_with_less_unassigned_neighbours(
            self,
            empty_values_counters: tuple[list[int], list[int], list[int]],
            cell1: Tuple[int, int],
            cell2: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Find the cell with less unassigned neighbours
        Args:
            empty_values_counters: tuple containing the number of empty values in each row, column and sub-square
            cell1: first cell
            cell2: second cell

        Returns:
            Tuple[int, int]: cell with less unassigned neighbours
        """
        row_index_1, col_index_1 = cell1
        row_index_2, col_index_2 = cell2

        sub_square_index_1 = self.get_sub_square_index(row_index_1, col_index_1)
        sub_square_index_2 = self.get_sub_square_index(row_index_2, col_index_2)

        row_empty_values_counter, col_empty_values_counter, sub_empty_values_counter = empty_values_counters

        cell1_empty_neighbours_count = (row_empty_values_counter[row_index_1] +
                                        col_empty_values_counter[col_index_1] +
                                        sub_empty_values_counter[sub_square_index_1])
        cell2_empty_neighbours_count = (row_empty_values_counter[row_index_2] +
                                        col_empty_values_counter[col_index_2] +
                                        sub_empty_values_counter[sub_square_index_2])

        if cell2_empty_neighbours_count <= cell1_empty_neighbours_count:
            return cell2

        return cell1

    def get_sub_square_index(self, row: int, col: int) -> int:
        """
        Get the index of the sub-square that contains the cell at the given row and column

        Args:
            row: row index
            col: column index

        Returns:
            int: index of the sub-square that contains the cell at the given row and column
        """
        n = len(self.board)
        return get_sub_square_index(n, row, col)

    def find_domains(self) -> List[List[Set[int]]]:
        """
        Find the domains of each cell in the board
        Returns:
            List[List[Set[int]]]: domains of each cell in the board
        """
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

    def is_valid_insertion(self, row: int, col: int, new_value: int) -> bool:
        """
        Check if the new value is valid in the given cell

        Args:
            row: row index
            col: column index
            new_value: value to be inserted

        Returns:
            bool: True if the new value is valid in the given cell, False otherwise
        """
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


def mask_board(original_board, p=0.75, seed=None):
    """
    Takes an n x n array and changes p percent of the values to 0.

    Args:
        original_board: list[list[int]] - an n x n array
        p: float - the proportion of values to change to 0 (default 0.75)
        seed: int or None - seed for the random number generator (default None)

    Returns:
        list[list[int]] - the modified n x n array
    """
    if seed is not None:
        random.seed(seed)

    board = copy.deepcopy(original_board)
    n = len(board)
    m = int(p * n * n)  # number of values to change to 0
    indices = random.sample(range(n * n), m)  # choose random indices to change
    for idx in indices:
        i, j = divmod(idx, n)  # calculate the row and column indices
        board[i][j] = 0  # set the value to 0
    return board


def main():
    """
    Main function. Used to test the SudokuSolver class.
    """
    board = mask_board(SIXTEEN_X_SIXTEEN_SOLVED)
    solver = SudokuSolver(board)
    solution_node = solver.solve()
    print(f"solution:\n{solution_node}")
    if solution_node:
        board = solution_node.board
        print(f"2D array: {board}")


def solve_with_brute_force(board):
    """
    Solve the given Sudoku board using brute force.

    Args:
        board: list[list[int]] - the Sudoku board to solve

    Returns:
        list[list[int]] - the solved Sudoku board

    Raises:
        PuzzleUnsolvedException: if the puzzle cannot be solved within the given time limit
    """
    solver = SudokuSolver(board)
    max_process_seconds = 10 * 60  # 5 minutes
    solution_node = solver.solve(max_process_seconds=max_process_seconds)

    if solution_node is None:
        raise PuzzleUnsolvedException()

    return solution_node.board


if __name__ == '__main__':
    main()
