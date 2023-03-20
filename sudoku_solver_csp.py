assignemnt = {(3, 5): 4}


class Assignments:
    def __init__(self):
        # key = (row_index, col_index, sub_square_index)
        self.data = {(3, 5, 0): 4}

    # key = (row_index, col_index, sub_square_index)
    def remove(self, key):
        pass

    # key = (row_index, col_index, sub_square_index)
    def add(self, key, value):
        pass

    def is_complete(self):
        pass

    def select_unassigned_variable(self, constraints):
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

    def find_order_domain_values_of_var(self, var, constraints):
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


class Constraints:
    def __init__(self):
        # key = (row_index, col_index, sub_square_index)
        # value = a set that representing the domain
        self.data = {(3, 5, 0): {1, 2, 3}}

    def add(self, inferences_to_add):
        pass

    def remove(self, inferences_to_remove):
        pass

    def is_consistent(self, value, var, constraints):
        pass


def new_backtrack(constraints, assignment):
    if assignment.is_complete():
        return assignment
    var = assignment.select_unassigned_variable(
        constraints)  # var = (row, col, sub_square)
    for value in assignment.find_order_domain_values_of_var(var, constraints):
        if assignment.is_consistent(value, var, constraints):
            assignment.add(var, value)
            # TODO: update follows
            inferences = inference(constraints, var, assignment)
            if inferences != "failure":
                csp.add_inference(inferences)
                result = backtrack(constraints, assignment)
                if result != "failure":
                    return result
                csp.remove_inference(inferences)
            del assignemnt[var]
    return "failure"


def backtrack(csp, assignment):
    if is_complete(assignment, csp):
        return assignment
    var = select_unassigned_variable(csp, assignment)  # var = (row, col)
    for value in order_domain_values(csp, var, assignment):
        if is_consistent(value, var, assignment, csp):
            assignment[var] = value
            inferences = inference(csp, var, assignment)
            if inferences != "failure":
                csp.add_inference(inferences)
                result = backtrack(csp, assignment)
                if result != "failure":
                    return result
                csp.remove_inference(inferences)
            del assignemnt[var]
    return "failure"


def is_complete(assignment, csp):
    pass


def select_unassigned_variable(csp, assignment):
    pass


def order_domain_values(csp, var, assignment):
    pass


def is_consistent(value, var, assignment, csp):
    pass


def inference(csp, var, assignment):
    pass


ROW = 0
COL = 0


class SudokuCSP:

    def __init__(self, initial_grid=None) -> None:
        self.variables = [(row, col) for row in range(9) for col in range(9)]
        self.domains = {(row, col): set(range(1, 10))
                        for row in range(9) for col in range(9)}
        self.constraints = self.build_constraints()

        # Set the initial values in the grid
        for row in range(9):
            for col in range(9):
                value = initial_grid[row][col]
                if value != 0:
                    self.assign_value(row, col, value)

    def build_constraints(self):
        def row_constraint(cell1, cell2):
            return cell1[ROW] == cell2[ROW]

        def col_constraint(cell1, cell2):
            return cell1[COL] == cell2[COL]

        def box_constraint(cell1, cell2):
            return (cell1[ROW] // 3 == cell2[ROW] // 3) and (cell1[COL] // 3 != cell2[COL] // 3)

        constraints = set()
        for cell1 in self.variables:
            for cell2 in self.variables:
                if cell1 != cell2:
                    if row_constraint(cell1, cell2) or col_constraint(cell1, cell2) or box_constraint(cell1, cell2):
                        constraints.add((cell1, cell2))
        return constraints

    def assign_value(self, row, col, value):
        if value in self.domains[(row, col)]:
            self.domains[(row, col)] = {value}

    def add_inferences(self, inferences):
        pass  # Implement adding inferences

    def remove_inferences(self, inferences):
        pass  # Implement removing inferences


csp = SudokuCSP()
