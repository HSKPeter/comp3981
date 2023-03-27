import copy

from assignment import Assignments
from constraint import Constraints

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


class Node:
    def __init__(self, assignment, constraint=None, cell_changed=None) -> None:
        self.assignment = assignment
        self.constraint = Constraints(assignment) if constraint is None else constraint
        self.cell_changed = cell_changed
        # self.infer(cell_changed)

        self.is_checked = False
        self.children_nodes = []

    def infer(self, cell_changed):
        inferences = self.assignment.infer(cell_changed, self.constraint)

        if inferences is not None:
            self.constraint.add_inferences(inferences)

    def __str__(self):
        return str(self.assignment)

    def get_first_unchecked_child(self):
        for node in self.children_nodes:
            if not node.is_checked:
                return node

    def is_solution(self):
        return self.assignment.is_complete()

    def copy_assignment(self):
        return copy.deepcopy(self.assignment)

    def copy_constraint(self):
        return copy.deepcopy(self.constraint)

    def expand(self):
        self.infer(self.cell_changed)

        most_preferred_cell = self.assignment.select_unassigned_cell(self.constraint)

        for value in self.assignment.find_ordered_domain_values(most_preferred_cell, self.constraint):
            assignment_copy = self.copy_assignment()

            assignment_copy.add(most_preferred_cell, value)
            self.children_nodes.insert(0, Node(assignment_copy, cell_changed=most_preferred_cell))

    def check(self):
        self.is_checked = True

    def copy(self):
        return copy.deepcopy(self)


def get_sub_square_index(n, row, col) -> int:
    sub_n = FLOOR_SQUARE_ROOTS[n]  # size of each sub-square
    sub_m = n // sub_n  # number of sub-squares in each row or column
    sub_row = row // sub_m
    sub_col = col // sub_n
    sub_square_index = sub_row * sub_m + sub_col

    # sub_square_index starting from 0, counting from top-left to bottom-right of the board
    return sub_square_index


class SudokuCspSolver:
    def __init__(self, board) -> None:
        assignment = Assignments(board)
        node = Node(assignment)
        self.stack = [node]

    def solve(self):
        while len(self.stack) > 0:

            current_node = self.stack[-1]

            if current_node.is_solution():
                return current_node

            # print(current_node)

            current_node.expand()
            next_node = current_node.get_first_unchecked_child()

            if next_node is None:
                current_node.check()
                self.stack.pop()
            else:
                self.stack.append(next_node)




# def solve_with_csp(board, recursion_limit=None):
#     assignments = Assignments(board)
#     constraints = Constraints(assignments)
#
#     if recursion_limit is not None:
#         sys.setrecursionlimit(recursion_limit)
#         print("Recursion limit: ", recursion_limit)
#
#     result = backtrack(constraints, assignments)
#
#     return result.to_2d_array()


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
    # NINE_X_NINE = [[0, 6, 2, 4, 9, 8, 5, 1, 3], [9, 3, 1, 2, 5, 6, 4, 8, 7], [4, 5, 8, 1, 3, 7, 2, 6, 9], [5, 1, 9, 7, 8, 2, 3, 4, 6], [6, 8, 7, 9, 4, 3, 1, 5, 2], [3, 2, 4, 5, 6, 1, 9, 7, 8], [2, 9, 6, 8, 1, 4, 7, 3, 5], [8, 4, 5, 3, 7, 9, 6, 2, 1], [1, 7, 3, 6, 2, 5, 8, 9, 4]]


    solver = SudokuCspSolver(NINE_X_NINE)
    result = solver.solve()
    print(result)


if __name__ == '__main__':
    main()
