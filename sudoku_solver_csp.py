import copy
import sys

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


def get_sub_square_index(n, row, col) -> int:
    sub_n = FLOOR_SQUARE_ROOTS[n]  # size of each sub-square
    sub_m = n // sub_n  # number of sub-squares in each row or column
    sub_row = row // sub_m
    sub_col = col // sub_n
    sub_square_index = sub_row * sub_m + sub_col
    # sub_square_index starting from 0, counting from top-left to bottom-right of the board
    return sub_square_index


class Assignments:
    def __init__(self, board):
        self.n = len(board)
        self.values = {(row, col, self.get_sub_square_index(
            row, col)): board[row][col] for row in range(self.n) for col in range(self.n)}
        self.all_arcs = self.find_all_arcs()

    # key = a tuple of (row_index, col_index, sub_square_index)
    def remove(self, key) -> None:
        pass

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
        all_arcs = set(self.all_arcs.values())
        for arc in all_arcs:
            cell_one, cell_two = arc
            if self.values[cell_one] == self.values[cell_two]:
                raise InvalidAssignmentException(
                    f"{cell_one} and {cell_two} were both assigned {self.values[cell_one]}")

    # constraints is a dict, where the key is a tuple of integers e.g. (row_index, col_index, sub_square_index) representing the cell position,
    # and the value would be a set of integers that represent the domain values of that cell
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
        best_cell = {
            "cell": (0, 0, 0),
            "heuristic_value": sys.maxsize
        }
        cells = list(self.values.keys())
        for cell in cells:
            if self.cell_is_empty(cell):
                continue
            domain_size = self.n - len(constraints.domains[cell])
            degree = self.find_degree(cell)
            heuristic_value = domain_size + degree
            if heuristic_value < best_cell["heuristic_value"]:
                best_cell["cell"] = cell
                best_cell["heuristic_value"] = heuristic_value
        # print("best_cell[\"cell\"]", best_cell["cell"])
        return best_cell["cell"]

    def cell_is_empty(self, cell) -> bool:
        return self.values[cell] == 0

    def find_degree(self, cell) -> int:
        degree = 0
        arcs = self.find_arcs(cell)
        # print(arcs)
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
        # print(constraints.domains)
        # print("Domain for cell ", cell_key, " ", domain_values)
        if len(domain_values) == 1:
            return domain_values

        def count_constraints(cell_key, value):
            count = 0
            for neighbor_key in self.find_cell_neighbours(cell_key):
                # If the value is in the domain of it's neighbours cells, then increment the count since this would now affect their domains
                if value in constraints.domains.get(neighbor_key):
                    count += 1
            return count

        domain_values.sort(key=lambda x: count_constraints(cell_key, x))

        return domain_values

    # cell is a tuple of integers e.g. (row_index, col_index, sub_square_index) representing the cell position
    # constraints is a dict, where the key is a tuple of integers e.g. (row_index, col_index, sub_square_index) representing the cell position,
    # and the value would be a set of integers that represent the domain values of that cell

    def infer(self, cell: (int, int, int), constraints: dict[(int, int): set[int]]) -> dict[(int, int): set[int]]:
        """
         Inference function: Use the Maintaining Arc Consistency (MAC) heuristic, which is based on the
         AC-3 algorithm. This helps ensure that the remaining variables maintain their arc consistency
         after assigning a value to the current variable.

        This function returns a dictionary where the keys are cells whose domain is updated based on the "cell" value
        If any of the domains reduce to none (size 0) then this function fails and returns None

        """
        constraints_copy = {key: value.copy()
                            for (key, value) in constraints.items()}
        constraints_copy[cell] = {self.values[cell]}
        initial_arcs = self.find_arcs(cell)
        queue = list(initial_arcs)

        while len(queue) > 0:
            cell_i, cell_j = queue.pop()
            has_revised, constraints_copy = self.revise(
                constraints_copy, cell_i, cell_j)
            if has_revised:
                if len(constraints_copy[cell_i]) == 0:
                    return None
                cell_neighbours = self.find_cell_neighbours(cell_i)
                cell_neighbours_excluding_cell_j = [
                    cell_neighbour for cell_neighbour in cell_neighbours if cell_neighbour != cell_j]
                for cell_neighbour in cell_neighbours_excluding_cell_j:
                    queue.append((cell_neighbour, cell_i))

        result_constraints = dict()
        old_constraints = dict()
        for key, value in constraints_copy.items():
            if constraints_copy[key] != constraints[key]:
                result_constraints[key] = value
                old_constraints[key] = constraints[key]

        return result_constraints, old_constraints

    def find_arcs(self, cell: (int, int, int)) -> {(int, int, int), (int, int, int)}:
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

    def print_board(self):
        sub_square_size = int(self.n ** 0.5)
        full_row = "+".join(["-" * (sub_square_size * 5 - 1)] * sub_square_size)

        for row in range(self.n):
            if row % sub_square_size == 0:
                print(full_row)
            row_str = ' |'
            for col in range(self.n):
                value = self.values[(row, col, self.get_sub_square_index(row, col))]
                if value == 0:
                    row_str += '__'
                else:
                    row_str += f'{value} '
                if (col + 1) % sub_square_size == 0:
                    row_str += '  |'
                row_str += "  "
            print(row_str)
            # if (row + 1) % sub_square_size == 0:
            #     print(full_row)

    @staticmethod
    def revise(constraints, cell_i, cell_j):
        constraints_copy = {key: value.copy()
                            for (key, value) in constraints.items()}
        revised = False
        for domain_value_i in constraints_copy[cell_i]:
            domain_of_cell_j = constraints[cell_j]
            filtered_domain_of_cell_j = {
                value for value in domain_of_cell_j if value != domain_value_i}
            if len(filtered_domain_of_cell_j) == 0:
                constraints[cell_i].remove(domain_value_i)
                revised = True
        return revised, constraints

    def copy(self):
        pass

    def to_2d_array(self):
        pass

    def get_sub_square_index(self, row, col) -> int:
        return get_sub_square_index(self.n, row, col)

    def is_consistent(self, cell, value, constraints):
        return True  # TODO: could remove this method if it is no longer needed eventually


class Constraints:
    def __init__(self, assignments: Assignments):
        # key = (row_index, col_index, sub_square_index)
        # Note: sub_square_index starting from 0, counting from top-left to bottom-right of the board
        # value = a set that representing the domain
        self.domains = dict()
        for cell_key, cell_value in assignments.values.items():
            if cell_value == 0:
                self.domains[cell_key] = set(range(1, 10))
            else:
                self.domains[cell_key] = {cell_value}

        self.domains_copy = copy.deepcopy(self.domains)

    def add_inferences(self, inferences_to_add):
        """
        Args:
            inferences_to_add: a dict where the key is a tuple of (row_index, col_index, sub_square_index),
            and the value is a set of numbers that represents the domain values
        """
        self.domains.update(inferences_to_add)

    def copy(self):
        return Constraints(self.data)

    def remove_inferences(self):
        self.domains = {key: value for (
            key, value) in self.domains_copy.items()}


def backtrack(constraints: Constraints, assignment: Assignments, depth: int = 0):
    if depth % 1000 == 0:
        print(depth)
        assignment.print_board()
    if assignment.is_complete():
        return assignment
    cell = assignment.select_unassigned_cell(
        constraints)  # cell = (row, col, sub_square)
    for value in assignment.find_ordered_domain_values(cell, constraints):
        if assignment.is_consistent(cell, value, constraints):
            assignment.add(cell, value)
            inferences, revert_inferences = assignment.infer(cell, constraints.domains)
            if inferences is not None:
                constraints.add_inferences(inferences)
                result = backtrack(constraints, assignment, depth + 1)
                if result is not None:
                    return result
                constraints.add_inferences(revert_inferences)
            assignment.remove(cell)
            # constraints = prev_constraints

    return None


def dev_backtrack(constraints: Constraints, assignment: Assignments):
    """
    TODO: this function is for dev and demo purpose only.  It would be removed in future.
    """
    print("Original assignments")
    print(assignment.values)
    print("Is complete")
    print(assignment.is_complete())
    print("Original constraints: ")
    print(constraints.domains)

    cell = (0, 1, 0)
    print("\nCell: ")
    print(cell)

    inferences, revert_inferences = assignment.infer(cell, constraints.domains)

    print("\nNew constraints inferred: ")
    print(inferences)

    print("These are updated from")
    print(revert_inferences)


def main():
    # NINE_X_NINE = [[0, 0, 3, 0, 2, 0, 6, 0, 0],
    #                [9, 0, 0, 3, 0, 5, 0, 0, 1],
    #                [0, 0, 1, 8, 0, 6, 4, 0, 0],
    #                [0, 0, 8, 1, 0, 2, 9, 0, 0],
    #                [7, 0, 0, 0, 0, 0, 0, 0, 8],
    #                [0, 0, 6, 7, 0, 8, 2, 0, 0],
    #                [0, 0, 2, 6, 0, 9, 5, 0, 0],
    #                [8, 0, 0, 2, 0, 3, 0, 0, 9],
    #                [0, 0, 5, 0, 1, 0, 3, 0, 0]]

    ONLY_EMPTY_VALUE = 0

    NINE_X_NINE = [[ONLY_EMPTY_VALUE, 6, 2, 4, 9, 8, 5, 1, 3], [9, 3, 1, 2, 5, 6, 4, 8, 7], [4, 5, 8, 1, 3, 7, 2, 6, 9], [5, 1, 9, 7, 8, 2, 3, 4, 6],
     [6, 8, 7, 9, 4, 3, 1, 5, 2], [3, 2, 4, 5, 6, 1, 9, 7, 8], [2, 9, 6, 8, 1, 4, 7, 3, 5], [8, 4, 5, 3, 7, 9, 6, 2, 1],
     [1, 7, 3, 6, 2, 5, 8, 9, 4]]
    test_assignments = Assignments(NINE_X_NINE)
    test_constraints = Constraints(test_assignments)
    print("recursion limit:", sys.setrecursionlimit(1000000))
    result = backtrack(test_constraints, test_assignments)
    print(result)
    # dev_backtrack(test_constraints, test_assignments)
    # print("Values: \n", test_assignments.values)
    # ordered_values = test_assignments.find_ordered_domain_values(
    #     (0, 0, 0), test_constraints)
    # print(ordered_values)


if __name__ == '__main__':
    main()
