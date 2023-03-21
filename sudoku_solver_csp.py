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




def backtrack(constraints: Constraints, assignment: Assignments):
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
                result = backtrack(constraints, assignment)
                if result is not None:
                    return result
                constraints.add_inferences(revert_inferences)
            assignment.remove(cell)
            # constraints = prev_constraints

    return None

def dev_stack(assignment):
    stack = []
    to_start = True

    while len(stack) > 0 or to_start:
        constraints, assignment = stack.pop()

        if assignment.is_complete():
            return assignment

        cell = assignment.select_unassigned_cell(constraints)

        for value in assignment.find_ordered_domain_values(cell, constraints):
            if assignment.is_consistent(cell, value, constraints):
                assignment.add(cell, value)
                inferences, revert_inferences = assignment.infer(cell, constraints.domains)
                if inferences is not None:
                    constraints.add_inferences(inferences)
                    stack.append(constraints, assignment)

        to_start = False



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
    NINE_X_NINE = [[0, 0, 3, 0, 2, 0, 6, 0, 0],
                   [9, 0, 0, 3, 0, 5, 0, 0, 1],
                   [0, 0, 1, 8, 0, 6, 4, 0, 0],
                   [0, 0, 8, 1, 0, 2, 9, 0, 0],
                   [7, 0, 0, 0, 0, 0, 0, 0, 8],
                   [0, 0, 6, 7, 0, 8, 2, 0, 0],
                   [0, 0, 2, 6, 0, 9, 5, 0, 0],
                   [8, 0, 0, 2, 0, 3, 0, 0, 9],
                   [0, 0, 5, 0, 1, 0, 3, 0, 0]]
    test_assignments = Assignments(NINE_X_NINE)
    test_constraints = Constraints(test_assignments)

    result = backtrack(test_constraints, test_assignments)
    print(result)
    # dev_backtrack(test_constraints, test_assignments)
    # print("Values: \n", test_assignments.values)
    # ordered_values = test_assignments.find_ordered_domain_values(
    #     (0, 0, 0), test_constraints)
    # print(ordered_values)


if __name__ == '__main__':
    main()
