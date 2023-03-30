import time
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Tuple
from sudoku_solver import mask_board


class SolverExecutionExpiredException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class PuzzleUnsolvedException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class InvalidAssignmentException(Exception):
    pass


class NeighborType(Enum):
    ROW = 1
    COL = 2
    SUB_SQUARE = 3


FLOOR_SQUARE_ROOTS = {
    9: 3,
    12: 3,
    16: 4,
    25: 5,
    100: 10
}


def get_sub_square_index(n, row, col) -> int:
    sub_n = FLOOR_SQUARE_ROOTS[n]  # size of each sub-square
    sub_m = n // sub_n  # number of sub-squares in each row or column
    sub_row = row // sub_m
    sub_col = col // sub_n
    sub_square_index = sub_row * sub_m + sub_col
    # sub_square_index starting from 0, counting from top-left to bottom-right of the board
    return sub_square_index


class SudokuSolverCsp:
    def __init__(self, board: List[List[int]], apply_rules=False) -> None:
        Node.n = len(board)
        domains = dict()
        for row in range(Node.n):
            for col in range(Node.n):
                key = (row, col, get_sub_square_index(Node.n, row, col))
                if board[row][col] != 0:
                    domains[key] = {board[row][col]}
                else:
                    domains[key] = set(range(1, Node.n + 1))

        first_node = Node(domains)
        self.stack = [first_node]
        self.reserved_stack = []

    def migrate_nodes_to_reserved_stack(self):
        for node in self.stack[1:]:
            node.reserve()
            self.reserved_stack.append(node)

        self.stack = self.stack[:1]

    def migrate_nodes_in_reserved_stack(self):
        self.stack = [Node.mark_node_as_unreserved(node) for node in self.reserved_stack]
        self.reserved_stack = []

    def solve(self, max_process_seconds=None):
        expiry_timestamp = (datetime.now() + timedelta(
            seconds=max_process_seconds)).timestamp() if max_process_seconds is not None else None
        i = 0
        timeout = 0

        while self.stack:
            if (expiry_timestamp is not None) and (time.time() >= expiry_timestamp):
                raise SolverExecutionExpiredException(f"No solution is found within {max_process_seconds} seconds")

            current_node = self.stack[-1]

            is_valid = current_node.do_forward_checking()

            if i % 10 == 0:
                print(i)
                if i % 100 == 0:
                    print(current_node)

            if not is_valid:
                current_node.check()
                self.stack.pop()
                continue

            if current_node.is_solution():
                return current_node

            current_node.expand()
            next_node = current_node.get_first_traversable_child()

            if next_node is None:
                current_node.check()
                self.stack.pop()
            else:
                self.stack.append(next_node)

            i += 1
            timeout += 1
            if timeout >= 5000:
                timeout = 0
                self.migrate_nodes_to_reserved_stack()

            if len(self.stack) == 0:
                self.migrate_nodes_in_reserved_stack()


class Node:
    all_arcs = None
    every_cell_neighbour = None
    n = None

    def __init__(self, domains,
                 assigned_cell=None,
                 new_value=None,
                 ) -> None:
        self.domains = {key: value.copy() for key, value in domains.items()}
        self.assigned_cell = assigned_cell
        if new_value is not None:
            self.domains[assigned_cell] = {new_value}
        self.children = []
        self.is_checked = False
        self.is_expanded = False
        self.is_reserved = False

        if Node.all_arcs is None:
            Node.all_arcs = self.find_all_arcs()
        if Node.every_cell_neighbour is None:
            Node.every_cell_neighbour = self.find_every_cell_neighbours()

    def get_arcs(self, cell: (int, int, int)) -> {(int, int, int), (int, int, int)}:
        return self.all_arcs[cell]

    def find_all_arcs(self):
        all_arcs = dict()
        for cell in self.domains.keys():
            cell_arcs = set()
            row_index, col_index, sub_square_index = cell
            for counter_cell in self.domains.keys():
                counter_cell_row_index, counter_cell_col_index, counter_cell_sub_square_index = counter_cell
                is_same_row = row_index == counter_cell_row_index
                is_same_col = col_index == counter_cell_col_index
                is_same_sub_square = sub_square_index == counter_cell_sub_square_index
                is_different_cell = cell != counter_cell

                if is_different_cell and (is_same_row or is_same_col or is_same_sub_square):
                    cell_arcs.add((cell, counter_cell))
            all_arcs[cell] = cell_arcs

        return all_arcs

    def find_every_cell_neighbours(self):
        return {cell: self.find_cell_neighbours(cell) for cell in self.domains.keys()}

    def find_cell_neighbours(self, cell):
        cell_neighbours = set()
        row_index, col_index, sub_square_index = cell
        for counter_cell in self.domains.keys():
            counter_cell_row_index, counter_cell_col_index, counter_cell_sub_square_index = counter_cell
            is_same_row = row_index == counter_cell_row_index
            is_same_col = col_index == counter_cell_col_index
            is_same_sub_square = sub_square_index == counter_cell_sub_square_index
            is_different_cell = cell != counter_cell

            if is_different_cell and (is_same_row or is_same_col or is_same_sub_square):
                cell_neighbours.add(counter_cell)

        return cell_neighbours

    def do_forward_checking(self):
        self.apply_hidden_single_rule()
        self.apply_naked_pair_rule()
        return self.infer()

    @staticmethod
    def mark_node_as_unreserved(node):
        node.unreserve()
        return node

    def reserve(self):
        self.is_reserved = True

    def unreserve(self):
        self.is_reserved = False

    def revise(self, cell_to_revise, cell_to_check):
        has_revised = False
        domain_of_cell_to_check: set = self.domains[cell_to_check]
        if len(domain_of_cell_to_check) == 1:
            # get one element from the set
            domain_value = next(iter(domain_of_cell_to_check))
            if domain_value in self.domains[cell_to_revise]:
                has_revised = True
                self.domains[cell_to_revise].remove(domain_value)
        return has_revised

    def infer(self):
        stack = []
        set_for_duplication_check = set()

        if self.assigned_cell is None:
            for _, arc_set in Node.all_arcs.items():
                for arc in arc_set:
                    stack.append(arc)
                    set_for_duplication_check.add(arc)
        else:
            for arc in self.get_arcs(self.assigned_cell):
                stack.append(arc)
                set_for_duplication_check.add(arc)

        while len(stack) > 0:
            arc_selected = stack.pop()
            set_for_duplication_check.remove(arc_selected)
            current_cell, other_cell = arc_selected
            has_revised = self.revise(cell_to_revise=other_cell, cell_to_check=current_cell)
            if has_revised:
                if len(self.domains[other_cell]) == 0:
                    return False
                else:
                    other_cell_neighbors = Node.every_cell_neighbour[other_cell]
                    for neighbor in other_cell_neighbors:
                        if neighbor != current_cell:
                            arc_to_prioritize = (other_cell, neighbor)
                            if arc_to_prioritize not in set_for_duplication_check:
                                stack.append(arc_to_prioritize)
                                set_for_duplication_check.add(arc_to_prioritize)
        return True

    def apply_naked_pair_rule(self):
        # Find if there is a pair of cells that have the same two values
        # If so, delete those values from the domain of the other cells in the same sub-square, row, and column
        for key, value in self.domains.items():
            for neighbor_key in Node.every_cell_neighbour[key]:
                neighbor_with_same_two_values = 0
                if self.domains.get(neighbor_key) == value:
                    neighbor_with_same_two_values += 1
                pair_found = neighbor_with_same_two_values == 2
                if pair_found:
                    for other_key in Node.every_cell_neighbour.get(key):
                        if other_key != neighbor_key:
                            self.domains[other_key].discard(value[0])
                            self.domains[other_key].discard(value[1])

    def find_cell_neighbours_by_type(self, cell: (int, int, int), neighbor_type: NeighborType):
        cell_neighbours = set()
        row_index, col_index, sub_square_index = cell

        for counter_cell in self.domains.keys():
            counter_cell_row_index, counter_cell_col_index, counter_cell_sub_square_index = counter_cell
            is_different_cell = cell != counter_cell

            if neighbor_type == NeighborType.ROW:
                is_same_group = row_index == counter_cell_row_index
            elif neighbor_type == NeighborType.COL:
                is_same_group = col_index == counter_cell_col_index
            elif neighbor_type == NeighborType.SUB_SQUARE:
                is_same_group = sub_square_index == counter_cell_sub_square_index
            else:
                raise ValueError(f"Invalid neighbor_type: {neighbor_type}")

            if is_different_cell and is_same_group:
                cell_neighbours.add(counter_cell)

        return cell_neighbours

    def apply_hidden_single_rule(self):
        # Implement the "hidden single" inference rule
        # If a region contains only one square which can hold a specific number, then that number must go into that square
        hidden_single_found = set()
        # Iterate through every assigned cell in the grid
        for cell_key, domain_values in self.domains.items():
            if len(domain_values) != 1:
                continue
            # Iterate through the neighbour of cell
            row_neighbours = self.find_cell_neighbours_by_type(cell_key, NeighborType.ROW)
            col_neighbours = self.find_cell_neighbours_by_type(cell_key, NeighborType.COL)
            sub_square_neighbours = self.find_cell_neighbours_by_type(cell_key, NeighborType.SUB_SQUARE)

            union_domain_values_of_row_neighbours = set()
            for neighbour in row_neighbours:
                union_domain_values_of_row_neighbours = union_domain_values_of_row_neighbours.union(
                    self.domains[neighbour])

            union_domain_values_of_col_neighbours = set()
            for neighbour in col_neighbours:
                union_domain_values_of_col_neighbours = union_domain_values_of_col_neighbours.union(
                    self.domains[neighbour])

            union_domain_values_of_sub_square_neighbours = set()
            for neighbour in sub_square_neighbours:
                union_domain_values_of_sub_square_neighbours = union_domain_values_of_sub_square_neighbours.union(
                    self.domains[neighbour])

            if len(union_domain_values_of_row_neighbours) == Node.n and len(
                    union_domain_values_of_col_neighbours) == Node.n and len(
                union_domain_values_of_sub_square_neighbours) == Node.n:
                continue

            #  Iterate through domain values of the cell
            for domain_value in self.domains[cell_key]:
                if (domain_value not in union_domain_values_of_row_neighbours) or (
                        domain_value not in union_domain_values_of_col_neighbours) or (
                        domain_value not in union_domain_values_of_sub_square_neighbours):
                    self.domains[cell_key] = {domain_value}
                    break

    def expand(self):
        if not self.is_expanded:
            # constraint_domain_copy = {key: value for key, value in self.constraints.domains.items()}

            cell_selected = self.select_unassigned_cell()

            for value in self.find_ordered_domain_values(cell_selected):
                new_node = Node(self.domains, cell_selected, value)
                self.children.append(new_node)

            self.is_expanded = True

    def cell_is_empty(self, cell: Tuple[int, int]):
        return len(self.domains[cell]) > 1

    def select_unassigned_cell(self) -> Tuple[int, int]:
        unassigned_cells = [cell for cell in self.domains.keys() if self.cell_is_empty(cell)]

        # Find the cells with the smallest domain size
        min_domain_size = min(len(self.domains[cell]) for cell in unassigned_cells)
        min_domain_cells = [cell for cell in unassigned_cells if len(self.domains[cell]) == min_domain_size]

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

    def find_degree(self, cell) -> int:
        degree = 0
        arcs = self.get_arcs(cell)
        for arc in arcs:
            other_cell = arc[1]
            if self.cell_is_empty(other_cell) == 0:
                degree += 1
        return degree

    def find_ordered_domain_values(self, cell_key):
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
        domain_values = list(self.domains[cell_key])
        if len(domain_values) == 1:
            return domain_values

        domain_values.sort(key=lambda x: self.count_constraints(cell_key, x))

        return domain_values

    def count_constraints(self, cell_key, value):
        count = 0
        for neighbor_key in self.every_cell_neighbour.get(cell_key):
            # If the value is in the domain of it's neighbours cells, then increment the count since this would now affect their domains
            if value in self.domains.get(neighbor_key):
                count += 1
        return count


    def check(self):
        self.is_checked = True

    def __str__(self):
        sub_square_size = int(self.n ** 0.5)
        full_row = "+".join(["-" * (sub_square_size * 5 - 1)] * sub_square_size)

        board_str = ''
        for row in range(self.n):
            if row % sub_square_size == 0:
                board_str += full_row + '\n'
            row_str = ' |'
            for col in range(self.n):
                domain_values = self.domains[(row, col, get_sub_square_index(Node.n, row, col))]
                value = 0
                if (len(domain_values) == 1):
                    value = next(iter(domain_values))
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

    def is_solution(self):
        """
        Check if this set of assignments is a solution to the problem (the whole board is filled and satisfies the constraints)
        """
        for domain_values in self.domains.values():
            if len(domain_values) != 1:
                return False
        all_arcs = set.union(*self.all_arcs.values())
        for arc in all_arcs:
            cell_one, cell_two = arc
            cell_one_value = next(iter(self.domains[cell_one]))
            cell_two_value = next(iter(self.domains[cell_two]))
            if cell_one_value == cell_two_value:
                raise InvalidAssignmentException(
                    f"{cell_one} and {cell_two} were both assigned the same value of {cell_one_value}\n{self}")
        return True

    def get_first_traversable_child(self):
        for node in self.children:
            if not node.is_checked and (node.is_reserved is False):
                return node
        return None

    def to_2d_array(self):
        two_d_array = []

        for i in range(Node.n):
            arr = list()
            for j in range(Node.n):
                arr.append(0)
            two_d_array.append(arr)

        for key, value in self.domains.items():
            row_index, col_index, _ = key
            two_d_array[row_index][col_index] = value

        return two_d_array


NINE_X_NINE_SOLVED = [[7, 6, 2, 4, 9, 8, 5, 1, 3],
                      [9, 3, 1, 2, 5, 6, 4, 8, 7],
                      [4, 5, 8, 1, 3, 7, 2, 6, 9],
                      [5, 1, 9, 7, 8, 2, 3, 4, 6],
                      [6, 8, 7, 9, 4, 3, 1, 5, 2],
                      [3, 2, 4, 5, 6, 1, 9, 7, 8],
                      [2, 9, 6, 8, 1, 4, 7, 3, 5],
                      [8, 4, 5, 3, 7, 9, 6, 2, 1],
                      [1, 7, 3, 6, 2, 5, 8, 9, 4]]

SIXTEEN_X_SIXTEEN_SOLVED = [[7, 4, 5, 11, 14, 9, 3, 12, 15, 2, 10, 1, 6, 8, 16, 13], [15, 9, 14, 6, 11, 10, 4, 5, 8, 13, 7, 16, 3, 2, 1, 12], [10, 16, 12, 1, 2, 13, 15, 8, 5, 3, 11, 6, 9, 14, 4, 7], [3, 13, 8, 2, 7, 6, 1, 16, 4, 9, 12, 14, 10, 15, 11, 5], [11, 14, 2, 10, 6, 12, 5, 13, 1, 15, 9, 4, 8, 7, 3, 16], [12, 5, 13, 9, 8, 3, 11, 1, 16, 7, 14, 2, 4, 10, 6, 15], [6, 8, 3, 16, 4, 7, 14, 15, 10, 5, 13, 12, 2, 1, 9, 11], [4, 15, 1, 7, 16, 2, 10, 9, 3, 11, 6, 8, 5, 12, 13, 14], [16, 1, 11, 3, 9, 5, 8, 6, 12, 14, 2, 7, 15, 13, 10, 4], [14, 12, 4, 5, 10, 15, 7, 11, 13, 8, 16, 9, 1, 3, 2, 6], [13, 2, 7, 8, 3, 1, 16, 14, 6, 10, 4, 15, 12, 11, 5, 9], [9, 10, 6, 15, 12, 4, 13, 2, 11, 1, 3, 5, 7, 16, 14, 8], [1, 6, 9, 12, 13, 11, 2, 7, 14, 4, 8, 3, 16, 5, 15, 10], [2, 3, 10, 13, 5, 16, 6, 4, 7, 12, 15, 11, 14, 9, 8, 1], [8, 11, 15, 4, 1, 14, 12, 3, 9, 16, 5, 10, 13, 6, 7, 2], [5, 7, 16, 14, 15, 8, 9, 10, 2, 6, 1, 13, 11, 4, 12, 3]]

TWENTY_FIVE_X_TWENTY_FIVE_SOLVED = [
    [23, 15, 16, 4, 18, 14, 2, 22, 5, 12, 20, 7, 8, 1, 10, 25, 11, 9, 17, 21, 6, 19, 24, 3, 13],
    [13, 19, 10, 1, 25, 23, 3, 7, 6, 17, 5, 2, 24, 18, 9, 20, 4, 14, 22, 15, 12, 16, 11, 21, 8],
    [8, 11, 21, 2, 6, 24, 15, 18, 1, 10, 12, 14, 17, 13, 19, 16, 3, 23, 5, 7, 22, 4, 25, 9, 20],
    [9, 22, 5, 12, 3, 4, 20, 21, 13, 8, 23, 16, 11, 15, 25, 2, 19, 6, 24, 18, 10, 7, 17, 14, 1],
    [17, 14, 24, 7, 20, 16, 11, 19, 9, 25, 22, 6, 4, 21, 3, 12, 13, 1, 10, 8, 23, 5, 15, 18, 2],
    [21, 2, 6, 8, 11, 10, 24, 1, 15, 18, 13, 12, 14, 19, 17, 3, 5, 7, 16, 23, 4, 22, 9, 20, 25],
    [5, 12, 3, 9, 22, 8, 4, 13, 20, 21, 15, 23, 16, 25, 11, 19, 24, 18, 2, 6, 7, 10, 14, 1, 17],
    [24, 7, 20, 17, 14, 25, 16, 9, 11, 19, 21, 22, 6, 3, 4, 13, 10, 8, 12, 1, 5, 23, 18, 2, 15],
    [16, 4, 18, 23, 15, 12, 14, 5, 2, 22, 1, 20, 7, 10, 8, 11, 17, 21, 25, 9, 19, 6, 3, 13, 24],
    [10, 1, 25, 13, 19, 17, 23, 6, 3, 7, 18, 5, 2, 9, 24, 4, 22, 15, 20, 14, 16, 12, 21, 8, 11],
    [3, 9, 12, 22, 5, 21, 13, 8, 4, 20, 11, 15, 25, 23, 16, 24, 18, 19, 6, 2, 17, 1, 7, 10, 14],
    [20, 17, 7, 14, 24, 19, 9, 25, 16, 11, 4, 21, 3, 22, 6, 10, 8, 13, 1, 12, 15, 2, 5, 23, 18],
    [25, 13, 1, 19, 10, 7, 6, 17, 23, 3, 24, 18, 9, 5, 2, 22, 15, 4, 14, 20, 11, 8, 16, 12, 21],
    [6, 8, 2, 11, 21, 18, 1, 10, 24, 15, 17, 13, 19, 12, 14, 5, 7, 3, 23, 16, 25, 20, 4, 22, 9],
    [18, 23, 4, 15, 16, 22, 5, 12, 14, 2, 8, 1, 10, 20, 7, 17, 21, 11, 9, 25, 24, 13, 19, 6, 3],
    [2, 6, 11, 21, 8, 1, 10, 15, 18, 24, 14, 19, 13, 17, 12, 23, 16, 5, 7, 3, 9, 25, 20, 4, 22],
    [12, 3, 22, 5, 9, 13, 8, 20, 21, 4, 16, 25, 15, 11, 23, 6, 2, 24, 18, 19, 14, 17, 1, 7, 10],
    [7, 20, 14, 24, 17, 9, 25, 11, 19, 16, 6, 3, 21, 4, 22, 1, 12, 10, 8, 13, 18, 15, 2, 5, 23],
    [4, 18, 15, 16, 23, 5, 12, 2, 22, 14, 7, 10, 1, 8, 20, 9, 25, 17, 21, 11, 3, 24, 13, 19, 6],
    [1, 25, 19, 10, 13, 6, 17, 3, 7, 23, 2, 9, 18, 24, 5, 14, 20, 22, 15, 4, 21, 11, 8, 16, 12],
    [22, 5, 9, 3, 12, 20, 21, 4, 8, 13, 25, 11, 23, 16, 15, 18, 6, 2, 19, 24, 1, 14, 10, 17, 7],
    [14, 24, 17, 20, 7, 11, 19, 16, 25, 9, 3, 4, 22, 6, 21, 8, 1, 12, 13, 10, 2, 18, 23, 15, 5],
    [19, 10, 13, 25, 1, 3, 7, 23, 17, 6, 9, 24, 5, 2, 18, 15, 14, 20, 4, 22, 8, 21, 12, 11, 16],
    [11, 21, 8, 6, 2, 15, 18, 24, 10, 1, 19, 17, 12, 14, 13, 7, 23, 16, 3, 5, 20, 9, 22, 25, 4],
    [15, 16, 23, 18, 4, 2, 22, 14, 12, 5, 10, 8, 20, 7, 1, 21, 9, 25, 11, 17, 13, 3, 6, 24, 19]]


def get_solved_board(size):
    if size == 9:
        return NINE_X_NINE_SOLVED
    elif size == 16:
        return SIXTEEN_X_SIXTEEN_SOLVED
    elif size == 25:
        return TWENTY_FIVE_X_TWENTY_FIVE_SOLVED

    raise ValueError("Invalid board size")


def main():
    board = get_solved_board(16)
    masked_board = mask_board(board)
    print(masked_board)
    start_time = time.time()
    sudoku_solver = SudokuSolverCsp(masked_board)
    result = sudoku_solver.solve()
    end_time = time.time()
    print("Solution")
    print(result)
    print(f"Solved in {end_time - start_time} seconds")


if __name__ == '__main__':
    main()