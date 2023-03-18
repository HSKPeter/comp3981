
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