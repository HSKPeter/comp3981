
assignemnt = {(3, 5): 4}


def backtrack(csp, assignment):
    if is_complete(assignment, csp):
        return assignment
    var = select_unassigned_variable(csp, assignment) # var = (row, col)
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
        self.variables = [(row, col) for row in range(9) for col in range (9)]
        self.domains = {(row, col): set(range(1, 10)) for row in range(9) for col in range(9)}
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