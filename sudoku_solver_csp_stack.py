from typing import List
from assignment import Assignments
from constraint import Constraints


class SolverExecutionExpiredException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class PuzzleUnsolvedException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)        


class SudokuSolver:
    def __init__(self, raw_board: List[List[int]]) -> None:
        self.board = raw_board
        self.stack = []

        assignments = Assignments(raw_board)
        constraints = Constraints(assignments)

        cell = assignments.select_unassigned_cell(constraints)

        for value in assignments.find_ordered_domain_values(cell, constraints):
            if assignments.is_consistent(cell, value, constraints):
                assignments.add(cell, value)
                inferences, revert_inferences = assignments.infer(cell, constraints.domains)
                if inferences is not None:
                    constraints.update_inferences(inferences)
                    self.stack.append(Node(assignments, constraints))
                    constraints.update_inferences(revert_inferences)
                    break
                assignments.remove(cell)

    def solve(self):
        stack_size = len(self.stack)

        while stack_size > 0:
            current_node = self.stack.pop()

            print(current_node)

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
    def __init__(self, assignments: Assignments, constraints: Constraints) -> None:
        self.assignments = assignments
        self.constraints = constraints
        self.children = []
        self.is_checked = False
        self.is_expanded = False

    def __str__(self):
        return str(self.assignments) + "\n"

    def expand(self):
        if not self.is_expanded:
            self.find_valid_children()
            self.is_expanded = True

    def check(self):
        self.is_checked = True

    def is_solution(self):
        return self.assignments.is_complete()

    def get_first_unchecked_child(self):
        for node in self.children:
            if not node.is_checked:
                return node
        return None

    def find_valid_children(self):
        cell = (0, 0, 0)
            # self.assignments.select_unassigned_cell(self.constraints)
        for value in self.assignments.find_ordered_domain_values(cell, self.constraints):
            if self.assignments.is_consistent(cell, value, self.constraints):
                self.assignments.add(cell, value)
                inferences, revert_inferences = self.assignments.infer(cell, self.constraints.domains)
                if inferences is not None:
                    self.constraints.update_inferences(inferences)
                    self.children.append(Node(self.assignments.copy(), self.constraints.copy()))
                    self.constraints.update_inferences(revert_inferences)
                self.assignments.remove(cell)


def main():
    NINE_X_NINE = [[0, 6, 2, 4, 9, 8, 5, 1, 3], [9, 3, 1, 2, 5, 6, 4, 8, 7], [4, 5, 8, 1, 3, 7, 2, 6, 9], [5, 1, 9, 7, 8, 2, 3, 4, 6], [6, 8, 7, 9, 4, 3, 1, 5, 2], [3, 2, 4, 5, 6, 1, 9, 7, 8], [2, 9, 6, 8, 1, 4, 7, 3, 5], [8, 4, 5, 3, 7, 9, 6, 2, 1], [1, 7, 3, 6, 2, 5, 8, 9, 4]]
        # [[0, 0, 3, 0, 2, 0, 6, 0, 0],
        #            [9, 0, 0, 3, 0, 5, 0, 0, 1],
        #            [0, 0, 1, 8, 0, 6, 4, 0, 0],
        #            [0, 0, 8, 1, 0, 2, 9, 0, 0],
        #            [7, 0, 0, 0, 0, 0, 0, 0, 8],
        #            [0, 0, 6, 7, 0, 8, 2, 0, 0],
        #            [0, 0, 2, 6, 0, 9, 5, 0, 0],
        #            [8, 0, 0, 2, 0, 3, 0, 0, 9],
        #            [0, 0, 5, 0, 1, 0, 3, 0, 0]]
    solver = SudokuSolver(NINE_X_NINE)
    solution_node = solver.solve()
    print(f"solution:\n{solution_node}")


if __name__ == '__main__':
    main()
