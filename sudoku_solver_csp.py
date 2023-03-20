ROW = 0
COL = 1

FLOOR_SQUARE_ROOTS = {
    9: 3,
    12: 3,
    16: 4,
    25: 5,
    100: 10
}


class Assignments:
    def __init__(self, board):
        self.n = len(board)
        # key = (row_index, col_index, sub_square_index)
        self.values = {(row, col): board[row][col] for row in range(self.n) for col in range(self.n)}

    def remove(self, key) -> None:
        pass

    # key = (row_index, col_index, sub_square_index)
    def add(self, key, value) -> None:
        pass

    def is_complete(self) -> bool:
        """
        check if this set of assignments is a solution to the problem (the whole board is filled and satisfies the constraints)
        """
        pass

    def select_unassigned_cell(self, constraints) -> (int, int):
        """
        Selecting unassigned variables: Use a combination of the Minimum Remaining Values (MRV) and Degree heuristics.
        MRV: Choose the variable with the fewest legal values remaining in its domain.
        Degree: Choose the variable involved in the highest number of constraints with other unassigned variables.

        Minimum Remaining Values (MRV):
T       he MRV heuristic chooses the unassigned variable with the fewest legal values remaining in its domain. In other words, it selects the variable that has the smallest number of possible values left to be assigned. The idea behind MRV is to select the variable that is most likely to cause a failure early on, thus pruning the search tree and reducing the overall search time.

        Degree:
        The Degree heuristic chooses the unassigned variable that is involved in the highest number of constraints with other unassigned variables. By selecting a variable with a high degree, you can potentially affect the most other unassigned variables in the problem. The Degree heuristic helps to resolve potential conflicts earlier in the search, which can also reduce the overall search time.

        While both MRV and Degree heuristics aim to improve the efficiency of the search process, they focus on different aspects of the problem. MRV looks at the remaining possibilities for a variable, while Degree looks at the relationships and constraints between unassigned variables. In some cases, it's beneficial to use a combination of both heuristics to select the most promising unassigned variable for the next step in the search process.
        """
        pass

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
        pass

    def infer(self, cell, constraints) -> dict[(int, int): set[int]]:
        """
         Inference function: Use the Maintaining Arc Consistency (MAC) heuristic, which is based on the 
         AC-3 algorithm. This helps ensure that the remaining variables maintain their arc consistency 
         after assigning a value to the current variable.

        This function returns a dictionary where the keys are are cells whose domain is updated based on the "cell" value
        If any of the domains reduce to none (size 0) then this function fails and returns None

        """
        pass

    def copy(self):
        pass

    def get_sub_square_index(self, row, col) -> int:
        n = self.n
        sub_n = FLOOR_SQUARE_ROOTS[n]  # size of each sub-square
        sub_m = n // sub_n  # number of sub-squares in each row or column
        sub_row = row // sub_m
        sub_col = col // sub_n
        sub_index = sub_row * sub_m + sub_col
        return sub_index


class Constraints:
    def __init__(self, assignments: Assignments):
        # key = (row_index, col_index, sub_square_index)
        # value = a set that representing the domain
        cells = assignments.values.keys()
        self.domains = {cell: set(range(1, 10)) for cell in cells}
        self.arcs = self.build_arcs(cells)

    def build_arcs(self, cells):
        def same_row(cell1, cell2):
            return cell1[ROW] == cell2[ROW]

        def same_column(cell1, cell2):
            return cell1[COL] == cell2[COL]

        def same_box(cell1, cell2):
            return (cell1[ROW] // 3 == cell2[ROW] // 3) and (cell1[COL] // 3 != cell2[COL] // 3)

        arcs = set()
        for cell1 in cells:
            for cell2 in cells:
                if cell1 != cell2:
                    if same_row(cell1, cell2) or same_column(cell1, cell2) or same_box(cell1, cell2):
                        arcs.add((cell1, cell2))
        return arcs

    def add_inferences(self, inferences_to_add):
        pass

    def remove_inferences(self, inferences_to_remove):
        pass

    def is_consistent(self, cell, value, constraints):
        pass


def backtrack(constraints: Constraints, assignment: Assignments):
    if assignment.is_complete():
        return assignment
    cell = assignment.select_unassigned_cell(constraints)  # cell = (row, col, sub_square)
    for value in assignment.find_ordered_domain_values(cell, constraints):
        if assignment.is_consistent(cell, value, constraints):
            assignment.add(cell, value)
            inferences = assignment.infer(cell, constraints)
            if inferences is not None:
                constraints.add_inferences(inferences)
                result = backtrack(constraints, assignment)
                if result is not None:
                    return result
                constraints.remove_inferences(inferences)
            assignment.remove(cell)
    return None


def main():
    pass


if __name__ == '__main__':
    main()
