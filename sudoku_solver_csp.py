import copy
import sys
import time

import sudoku_solver

ROW = 0
COL = 1

FLOOR_SQUARE_ROOTS = {
    9: 3,
    12: 3,
    16: 4,
    25: 5,
    100: 10
}


class InvalidAssignmentException(Exception):
    pass


class EmptyDomainException(Exception):
    pass


def get_sub_square_index(n, row, col) -> int:
    sub_n = FLOOR_SQUARE_ROOTS[n]  # size of each sub-square
    sub_m = n // sub_n  # number of sub-squares in each row or column
    sub_row = row // sub_m
    sub_col = col // sub_n
    sub_square_index = sub_row * sub_m + sub_col
    # sub_square_index starting from 0, counting from top-left to bottom-right of the board
    return sub_square_index


class ArcsCollection:
    def __init__(self, initial_arcs):
        self._arcs = set(initial_arcs)
        self._priority_arcs_stack = list()

    def __len__(self):
        return len(self._arcs) + len(self._priority_arcs_stack)

    def pop(self):
        if len(self._priority_arcs_stack) > 0:
            return self._priority_arcs_stack.pop()

        return self._arcs.pop()

    def add_priority_arc(self, arc):
        if arc in self._arcs:
            self._arcs.remove(arc)

        self._priority_arcs_stack.append(arc)


class Assignments:
    def __init__(self, board):
        self.n = len(board)
        self.values = {(row, col, self.get_sub_square_index(
            row, col)): board[row][col] for row in range(self.n) for col in range(self.n)}
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


class Constraints:
    def __init__(self, assignments: Assignments):
        # key = (row_index, col_index, sub_square_index)
        # Note: sub_square_index starting from 0, counting from top-left to bottom-right of the board
        # value = a set that representing the domain
        self.domains = dict()
        for cell_key, cell_value in assignments.values.items():
            if cell_value == 0:
                self.domains[cell_key] = set(range(1, assignments.n + 1))
            else:
                self.domains[cell_key] = {cell_value}

        self.domains = self.trim_domains(self.domains, assignments.n)

    @staticmethod
    def trim_domains(domains, n):
        updated_domains = domains.copy()
        base = FLOOR_SQUARE_ROOTS[n]
        for cell, domain in domains.items():
            if len(domain) == 1:
                filled_value = next(iter(domain))
                row, col, subsquare = cell

                # Remove filled value from row domains
                for r in range(n):
                    if r != row:
                        updated_domains[(r, col, base * (r // base) + col // base)].discard(filled_value)

                # Remove filled value from column domains
                for c in range(n):
                    if c != col:
                        updated_domains[(row, c, base * (row // base) + c // base)].discard(filled_value)

                # Remove filled value from subsquare domains
                subsquare_row_start = base * (row // base)
                subsquare_col_start = base * (col // base)
                for r in range(subsquare_row_start, subsquare_row_start + 3):
                    for c in range(subsquare_col_start, subsquare_col_start + 3):
                        if r != row and c != col:
                            updated_domains[(r, c, base * (r // base) + c // base)].discard(filled_value)

        return updated_domains

    def add_inferences(self, inferences_to_add):
        """
        Args:
            inferences_to_add: a dict where the key is a tuple of (row_index, col_index, sub_square_index),
            and the value is a set of numbers that represents the domain values
        """
        self.domains.update(inferences_to_add)


backtrack_counter = 0
target_depth = 0

def backtrack(constraints: Constraints, assignment: Assignments, depth: int = 0, mute=True):
    global backtrack_counter
    global target_depth
    if backtrack_counter > 1000:
        if target_depth == 0:
            target_depth = depth/2
        if depth > target_depth:
            return None
        else:
            target_depth = 0
            backtrack_counter = 0
    backtrack_counter += 1

    if depth % 10 == 0 and not mute:
        print(f"depth {depth}    backtrack_counter {backtrack_counter}")
        print(assignment)
        print()
    if assignment.is_complete():
        if not mute:
            print("SOLUTION")
            print(assignment)
        return assignment
    cell = assignment.select_unassigned_cell(constraints)  # cell = (row, col, sub_square)
    for value in assignment.find_ordered_domain_values(cell, constraints):
        assignment.add(cell, value)
        inference_results = assignment.infer(cell, constraints.domains)
        if inference_results is not None:
            inferences, revert_inferences = inference_results
            constraints.add_inferences(inferences)
            result = backtrack(constraints, assignment, depth + 1, mute)
            if result is not None:
                return result
            constraints.add_inferences(revert_inferences)
        assignment.remove(cell)

    return None


def solve_with_csp(board, recursion_limit=None):
    assignments = Assignments(board)
    constraints = Constraints(assignments)

    if recursion_limit is not None:
        sys.setrecursionlimit(recursion_limit)
        print("Recursion limit: ", recursion_limit)

    result = backtrack(constraints, assignments)

    return result.to_2d_array()


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

    sys.setrecursionlimit(1000000)

    board = sudoku_solver.mask_board(sudoku_solver.TWENTY_FIVE_X_TWENTY_FIVE)
    # first algo
    assignments = Assignments(board)
    constraints = Constraints(assignments)
    start_time = time.time()
    result = backtrack(constraints, assignments, mute=False)
    end_time = time.time()
    print("Solution")
    print(result)
    print(f"Solved in {end_time - start_time} seconds")
    # print(result.to_2d_array())


if __name__ == '__main__':
    main()
