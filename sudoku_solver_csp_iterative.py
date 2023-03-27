import copy
from typing import List
import random
from datetime import datetime, timedelta
import time
from sudoku_solver_csp import InvalidAssignmentException, EmptyDomainException, Constraints, get_sub_square_index

NINE_X_NINE = [[0, 0, 3, 0, 2, 0, 6, 0, 0], [9, 0, 0, 3, 0, 5, 0, 0, 1], [0, 0, 1, 8, 0, 6, 4, 0, 0],
               [0, 0, 8, 1, 0, 2, 9, 0, 0], [
                   7, 0, 0, 0, 0, 0, 0, 0, 8], [0, 0, 6, 7, 0, 8, 2, 0, 0], [0, 0, 2, 6, 0, 9, 5, 0, 0],
               [8, 0, 0, 2, 0, 3, 0, 0, 9], [0, 0, 5, 0, 1, 0, 3, 0, 0]]
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

FLOOR_SQUARE_ROOTS = {
    9: 3,
    12: 3,
    16: 4,
    25: 5,
    100: 10
}


class Assignments:
    def __init__(self, n, values):
        self.n = n
        self.values = values
        self.all_arcs = self.find_all_arcs()
        self.every_cell_neighbour = self.find_every_cell_neighbours()

    # key = a tuple of (row_index, col_index, sub_square_index)
    def remove(self, key) -> None:
        self.values[key] = 0

    # key = (row_index, col_index, sub_square_index)
    def add(self, key, value) -> None:
        self.values[key] = value

    def is_complete(self) -> bool:
        """
        check if this set of assignments is a solution to the problem (the whole board is filled and satisfies the constraints)
        """
        for value in self.values.values():
            if value == 0:
                return False
        all_arcs = set.union(*self.all_arcs.values())
        for arc in all_arcs:
            cell_one, cell_two = arc
            if self.values[cell_one] == self.values[cell_two]:
                raise InvalidAssignmentException(
                    f"{cell_one} and {cell_two} were both assigned {self.values[cell_one]}\n{self}")
        return True

    def select_unassigned_cell(self, constraints) -> (int, int):
        """
        Selecting unassigned variables: Use a combination of the Minimum Remaining Values (MRV) and Degree heuristics.
        MRV: Choose the variable with the fewest legal values remaining in its domain.
        Degree: Choose the variable involved in the highest number of constraints with other unassigned variables.

        Minimum Remaining Values (MRV):
        The MRV heuristic chooses the unassigned variable with the fewest legal values remaining in its domain. In other words, it selects the variable that has the smallest number of possible values left to be assigned. The idea behind MRV is to select the variable that is most likely to cause a failure early on, thus pruning the search tree and reducing the overall search time.

        Degree:
        The Degree heuristic chooses the unassigned variable that is involved in the highest number of constraints with other unassigned variables. By selecting a variable with a high degree, you can potentially affect the most other unassigned variables in the problem. The Degree heuristic helps to resolve potential conflicts earlier in the search, which can also reduce the overall search time.

        While both MRV and Degree heuristics aim to improve the efficiency of the search process, they focus on different aspects of the problem. MRV looks at the remaining possibilities for a variable, while Degree looks at the relationships and constraints between unassigned variables. In some cases, it's beneficial to use a combination of both heuristics to select the most promising unassigned variable for the next step in the search process.
        """
        unassigned_cells = [cell for cell in self.values.keys() if self.cell_is_empty(cell)]

        # Find the cells with the smallest domain size
        min_domain_size = min(len(constraints.domains[cell]) for cell in unassigned_cells)
        min_domain_cells = [cell for cell in unassigned_cells if len(constraints.domains[cell]) == min_domain_size]

        if len(min_domain_cells) == 1:
            return min_domain_cells[0]

        # If there are ties for the smallest domain size, use the degree as a tie-breaker
        max_degree_cell = min_domain_cells[0]
        max_degree = self.find_degree(max_degree_cell)

        for cell in min_domain_cells[1:]:
            cell_degree = self.find_degree(cell)
            if cell_degree > max_degree:
                max_degree = cell_degree
                max_degree_cell = cell

        return max_degree_cell

    def cell_is_empty(self, cell) -> bool:
        return self.values[cell] == 0

    def find_degree(self, cell) -> int:
        degree = 0
        arcs = self.get_arcs(cell)
        for arc in arcs:
            other_cell = arc[1]
            if self.cell_is_empty(other_cell) == 0:
                degree += 1
        return degree

    # cell_key is a tuple of integers e.g. (row_index, col_index, sub_square_index) representing the cell position
    # constraints is a dict, where the key is a tuple of integers e.g. (row_index, col_index, sub_square_index) representing the cell position,
    # and the value would be a set of integers that represent the domain values of that cell
    def find_ordered_domain_values(self, cell_key, constraints) -> list[int]:
        """
        Ordering values of a variable: Use the Least Constraining Value (LCV) heuristic,
        which selects the value that imposes the fewest constraints on the remaining variables.
        The Least Constraining Value (LCV) heuristic is used to order the values of a variable when attempting to assign a value during the search process in a Constraint Satisfaction Problem (CSP). The LCV heuristic aims to minimize the impact of the current assignment on the future assignments of other variables.

        The main idea behind the LCV heuristic is to choose a value that imposes the fewest constraints on the remaining unassigned variables. By doing so, you keep more options open for the remaining variables, which can potentially result in fewer backtracks and a more efficient search process.

        To implement the LCV heuristic, you need to perform the following steps:

        1. For the current variable you are assigning a value to, evaluate each possible value in its domain.
        2. For each value, count the number of constraints it imposes on the remaining unassigned variables. This typically involves counting how many legal values are eliminated from the domains of neighboring 3. unassigned variables if the current value is assigned to the current variable.
        4. Order the values by the number of constraints they impose, from least constraining to most constraining.
        3. Assign the values to the current variable in the order determined by the LCV heuristic.
        By using the LCV heuristic, you can increase the likelihood of finding a solution without the need for excessive backtracking. This can result in a more efficient search process and, ultimately, a faster solution to the CSP.
        """
        # cell_key is a tuple of three integers
        # constraints is a Constraints object
        domain_values = list(constraints.domains.get(cell_key))
        if len(domain_values) == 1:
            return domain_values

        def count_constraints(cell_key, value):
            count = 0
            for neighbor_key in self.every_cell_neighbour.get(cell_key):
                # If the value is in the domain of it's neighbours cells, then increment the count since this would now affect their domains
                if value in constraints.domains.get(neighbor_key):
                    count += 1
            return count

        domain_values.sort(key=lambda x: count_constraints(cell_key, x))

        return domain_values

    # # cell is a tuple of integers e.g. (row_index, col_index, sub_square_index) representing the cell position
    # # constraints is a dict, where the key is a tuple of integers e.g. (row_index, col_index, sub_square_index) representing the cell position,
    # # and the value would be a set of integers that represent the domain values of that cell
    def infer(self, assigned_cell: (int, int, int), constraints: dict[(int, int, int): set[int]]) -> dict[(int, int, int): set[int]]:
        """
         Inference function: Use the Maintaining Arc Consistency (MAC) heuristic, which is based on the
         AC-3 algorithm. This helps ensure that the remaining variables maintain their arc consistency
         after assigning a value to the current variable.

        This function returns a dictionary where the keys are cells whose domain is updated based on the "cell" value
        If any of the domains reduce to none (size 0) then this function fails and returns None

        """
        assigned_value = self.values[assigned_cell]
        constraints_copy = {key: value.copy()
                            for (key, value) in constraints.items()}
        constraints_copy[assigned_cell] = {assigned_value}

        queue = list(self.get_arcs(assigned_cell))

        while len(queue) > 0:
            current_cell, other_cell = queue.pop()
            has_revised, constraints_copy = self.revise(constraints_copy, cell_to_revise=other_cell, cell_to_check=current_cell)
            if has_revised:
                if len(constraints_copy[other_cell]) == 0:
                    return None
                other_cell_neighbors = self.every_cell_neighbour[other_cell]
                for neighbor in other_cell_neighbors:
                    if neighbor != current_cell:
                        arc_to_prioritize = (other_cell, neighbor)
                        if arc_to_prioritize in queue:
                            queue.remove(arc_to_prioritize)
                        queue.append(arc_to_prioritize)

        result_constraints = dict()
        old_constraints = dict()
        for key, value in constraints_copy.items():
            if constraints_copy[key] != constraints[key]:
                if len(value) == 0:
                    raise EmptyDomainException(f"{key}: {value}")
                result_constraints[key] = value
                old_constraints[key] = constraints[key]

        return result_constraints, old_constraints

    @staticmethod
    def revise(constraints, cell_to_revise, cell_to_check):
        has_revised = False
        domain_of_cell_to_check: set = constraints[cell_to_check]
        if len(domain_of_cell_to_check) == 1:
            # get one element from the set
            domain_value = next(iter(domain_of_cell_to_check))
            for value_in_cell_to_revise in list(constraints[cell_to_revise]):
                if domain_value == value_in_cell_to_revise:
                    has_revised = True
                    constraints[cell_to_revise].remove(value_in_cell_to_revise)
        return has_revised, constraints

    def get_arcs(self, cell: (int, int, int)) -> {(int, int, int), (int, int, int)}:
        return self.all_arcs[cell]

    # (int, int, int) = cell
    # ((int, int, int), (int, int, int)) = binary arc
    # set(((int, int, int), (int, int, int))) = set of binary arc
    def compute_arcs(self, cell: (int, int, int)) -> set[((int, int, int), (int, int, int))]:
        arcs = set()
        row_index, col_index, sub_square_index = cell
        for counter_cell in self.values.keys():
            counter_cell_row_index, counter_cell_col_index, counter_cell_sub_square_index = counter_cell
            is_same_row = row_index == counter_cell_row_index
            is_same_col = col_index == counter_cell_col_index
            is_same_sub_square = sub_square_index == counter_cell_sub_square_index
            is_different_cell = cell != counter_cell

            if is_different_cell and (is_same_row or is_same_col or is_same_sub_square):
                arcs.add((cell, counter_cell))

        return arcs

    def find_all_arcs(self) -> dict[(int, int, int), set]:
        all_arcs = dict()
        for cell in self.values.keys():
            arcs = self.compute_arcs(cell)
            all_arcs[cell] = arcs
        return all_arcs

    def find_every_cell_neighbours(self):
        return {cell: self.find_cell_neighbours(cell) for cell in self.values.keys()}

    def find_cell_neighbours(self, cell: (int, int, int)):
        cell_neighbours = set()
        row_index, col_index, sub_square_index = cell
        for counter_cell in self.values.keys():
            counter_cell_row_index, counter_cell_col_index, counter_cell_sub_square_index = counter_cell
            is_same_row = row_index == counter_cell_row_index
            is_same_col = col_index == counter_cell_col_index
            is_same_sub_square = sub_square_index == counter_cell_sub_square_index
            is_different_cell = cell != counter_cell

            if is_different_cell and (is_same_row or is_same_col or is_same_sub_square):
                cell_neighbours.add(counter_cell)

        return cell_neighbours

    def __str__(self):
        sub_square_size = int(self.n ** 0.5)
        full_row = "+".join(["-" * (sub_square_size * 5 - 1)] * sub_square_size)

        board_str = ''
        for row in range(self.n):
            if row % sub_square_size == 0:
                board_str += full_row + '\n'
            row_str = ' |'
            for col in range(self.n):
                value = self.values[(row, col, self.get_sub_square_index(row, col))]
                if value == 0:
                    row_str += '__'
                else:
                    row_str += f'{value} ' if value < 10 else f'{value}'
                if (col + 1) % sub_square_size == 0:
                    row_str += '  |'
                row_str += "  "
            board_str += row_str + '\n'
        board_str += full_row

        return board_str + '\n'

    def get_sub_square_index(self, row, col) -> int:
        return get_sub_square_index(self.n, row, col)

    def to_2d_array(self):
        two_d_array = list()
        for i in range(self.n):
            arr = list()
            for j in range(self.n):
                arr.append(0)
            two_d_array.append(arr)

        for key, value in self.values.items():
            row_index, col_index, _ = key
            two_d_array[row_index][col_index] = value

        return two_d_array



class SolverExecutionExpiredException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class PuzzleUnsolvedException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class SudokuSolver:
    def __init__(self, raw_board: List[List[int]]) -> None:
        # self.board = raw_board
        self.n = len(raw_board)
        values = {(row, col, get_sub_square_index(self.n, row, col)): raw_board[row][col] for row in range(self.n) for col in range(self.n)}
        self.stack = [Node(self.n, values)]

    def solve(self):
        stack_size = len(self.stack)
        while stack_size > 0:
            current_node = self.stack[-1]
            if current_node.is_solution():
                return current_node

            current_node.expand()
            next_node = current_node.get_first_unchecked_child()

            if next_node is None:
                current_node.check()
                self.stack.remove(current_node)
            else:
                self.stack.append(next_node)


class Node:
    def __init__(self, n, values) -> None:
        self.n = n
        self.assignments = Assignments(n, values)
        self.constraints = Constraints(self.assignments)
        self.children = []
        self.is_checked = False
        self.is_expanded = False

    def expand(self):
        if not self.is_expanded:
            cell_selected = self.assignments.select_unassigned_cell(self.constraints)
            for value in self.assignments.find_ordered_domain_values(cell_selected, self.constraints):
                values_copy = copy.deepcopy(self.assignments.values)
                values_copy[cell_selected] = value
                self.children.append(Node(self.n, values_copy))
            self.is_expanded = True

    def check(self):
        self.is_checked = True

    def __str__(self):
        return str(self.assignments)

    def is_solution(self):
        return self.assignments.is_complete()

    def get_first_unchecked_child(self):
        for node in self.children:
            if not node.is_checked:
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


def main():
    NINE_X_NINE = [[0, 0, 3, 0, 2, 0, 6, 0, 0],
                   [9, 0, 0, 3, 0, 5, 0, 0, 1],
                   [0, 0, 1, 8, 0, 6, 4, 0, 0],
                   [0, 0, 8, 1, 0, 2, 9, 0, 0],
                   [7, 0, 0, 0, 0, 0, 0, 0, 8],
                   [0, 0, 6, 7, 0, 8, 2, 0, 0],
                   [0, 0, 2, 6, 0, 9, 5, 0, 0],
                   [8, 0, 0, 2, 0, 3, 0, 0, 9],
                   [0, 0, 5, 0, 1, 0, 3, 0, 0]]


    start_time = time.time()
    sudoku_solver = SudokuSolver(NINE_X_NINE)
    result = sudoku_solver.solve()
    end_time = time.time()
    print("Solution")
    print(result)
    print(f"Solved in {end_time - start_time} seconds")


if __name__ == '__main__':
    main()
