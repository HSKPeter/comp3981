import copy
import time
from sudoku_solver import mask_board
from datetime import datetime, timedelta
import heapq

ROW = 0
COL = 1

FLOOR_SQUARE_ROOTS = {
    9: 3,
    12: 3,
    16: 4,
    25: 5,
    100: 10
}



class PriorityQueue:
    def __init__(self, queue):
        self._queue = queue
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

    def empty(self):
        return len(self._queue) == 0

class InvalidAssignmentException(Exception):
    pass


class Node:
    def __init__(self, n, domains, assigned_cell=None) -> None:
        self.n = n
        self.domains = domains
        self.assigned_cell = assigned_cell
        self.all_arcs = self.find_all_arcs()
        self.children_nodes = []
        self.is_checked = False
        self.has_expanded = False
        # self.is_invalid = False

        # for cell_key, cell_domain in self.domains.items():
        #     if cell_domain is None:
        #         cell_neighbours = self.find_cell_neighbours(cell_key)
        #         restricted_domain_values = set()
        #         for cell_neighbour in cell_neighbours:
        #             cell_neighbour_domain = self.domains.get(cell_neighbour)
        #             if cell_neighbour_domain is not None:
        #                 restricted_domain_values = restricted_domain_values.union(cell_neighbour_domain)
        #         self.domains[cell_key] = set(range(1, self.n + 1)).difference(restricted_domain_values)
        #         if len(self.domains[cell_key]) == 0:
        #             self.is_invalid = True

    # def determine_cell_domain(self, cell_key):
    #     all_possible_values = set(range(1, self.n + 1))
    #     for neighbor_key in self.find_cell_neighbours(cell_key):
    #         neighbor_domain = self.domains.get(neighbor_key)
    #         all_possible_values = all_possible_values.difference(neighbor_domain)
    #     return all_possible_values

    def find_all_arcs(self) -> dict[(int, int, int), set]:
        all_arcs = dict()
        for cell in self.domains.keys():
            arcs = self.compute_arcs(cell)
            all_arcs[cell] = arcs
        return all_arcs

    def compute_arcs(self, cell: (int, int, int)) -> set[((int, int, int), (int, int, int))]:
        arcs = set()
        row_index, col_index, sub_square_index = cell
        for counter_cell in self.domains.keys():
            counter_cell_row_index, counter_cell_col_index, counter_cell_sub_square_index = counter_cell
            is_same_row = row_index == counter_cell_row_index
            is_same_col = col_index == counter_cell_col_index
            is_same_sub_square = sub_square_index == counter_cell_sub_square_index
            is_different_cell = cell != counter_cell

            if is_different_cell and (is_same_row or is_same_col or is_same_sub_square):
                arcs.add((cell, counter_cell))

        return arcs

    def has_cell_with_empty_domain(self):
        for cell_key, cell_domain in self.domains.items():
            if len(cell_domain) == 0:
                return True
        return False

    def trim_domains(self):
        for cell_key, cell_domain in self.domains.items():
            if len(cell_domain) == 1:
                domain_value = next(iter(cell_domain))
                cell_neighbours = self.find_cell_neighbours(cell_key)
                for cell_neighbour in cell_neighbours:
                    cell_neighbour_domain = self.domains[cell_neighbour]
                    if domain_value in cell_neighbour_domain:
                        cell_neighbour_domain.remove(domain_value)
            # if cell_domain is None:
            #     cell_neighbours = self.find_cell_neighbours(cell_key)
            #     restricted_domain_values = set()
            #     for cell_neighbour in cell_neighbours:
            #         cell_neighbour_domain = self.domains.get(cell_neighbour)
            #         if cell_neighbour_domain is not None:
            #             restricted_domain_values = restricted_domain_values.union(cell_neighbour_domain)
            #     self.domains[cell_key] = set(range(1, self.n + 1)).difference(restricted_domain_values)
            #     if len(self.domains[cell_key]) == 0:
            #         self.is_invalid = True

    def infer(self):
        # data_structure = set()
        data_structure = []
        # heapq.heapify(data_structure)

        domains_copy = {key: value.copy() for (key, value) in self.domains.items()}

        # if self.assigned_cell is None:
        for _, arc_set in self.all_arcs.items():
            for arc in arc_set:
                data_structure.append(arc)

        while len(data_structure) > 0:
            arc = data_structure.pop()
            cell_i, cell_j = arc
            has_revised, domains_copy = self.revise(
                domains_copy, cell_i, cell_j)
            if has_revised:
                if len(domains_copy[cell_i]) == 0:
                    return False
                cell_neighbours = self.find_cell_neighbours(cell_i)
                cell_neighbours_excluding_cell_j = [
                    cell_neighbour for cell_neighbour in cell_neighbours if cell_neighbour != cell_j]
                assigned_value_of_cell_i = next(iter(domains_copy[cell_i])) if len(domains_copy[cell_i]) == 1 else None
                for cell_neighbour in cell_neighbours_excluding_cell_j:
                    if assigned_value_of_cell_i is not None:
                        cell_neighbour_domain = domains_copy[cell_neighbour]
                        if assigned_value_of_cell_i in cell_neighbour_domain:
                            cell_neighbour_domain.remove(assigned_value_of_cell_i)
                    arc_to_add = (cell_neighbour, cell_i)
                    data_structure.append(arc_to_add)

        self.domains.update(domains_copy)
        return True

    @staticmethod
    def revise(domains, cell_i, cell_j):
        domains_copy = {key: value.copy() for (key, value) in domains.items()}
        revised = False
        for domain_value_i in domains_copy[cell_i]:
            domain_of_cell_j = domains[cell_j]
            filtered_domain_of_cell_j = {
                value for value in domain_of_cell_j if value != domain_value_i}
            if len(filtered_domain_of_cell_j) == 0:
                domains[cell_i].remove(domain_value_i)
                revised = True
        return revised, domains

    def find_cell_neighbours(self, cell: (int, int, int)):
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

    def __str__(self):
        return str(self.domains)

    def get_first_unchecked_child(self):
        for node in self.children_nodes:
            if not node.is_checked:
                return node

    def is_solution(self):
        for value in self.domains.values():
            if len(value) != 1:
                return False
        all_arcs = set.union(*self.all_arcs.values())
        for arc in all_arcs:
            cell_one, cell_two = arc
            if self.domains[cell_one] == self.domains[cell_two]:
                raise InvalidAssignmentException(
                    f"{cell_one} and {cell_two} were both assigned {self.domains[cell_one]}\n{self}")
        return True

    def expand(self):
        if self.has_expanded is False:
            most_preferred_cell = self.select_unassigned_cell()

            for value in self.find_ordered_domain_values(most_preferred_cell):
                domain_copy = {key: value.copy() for (key, value) in self.domains.items()}
                domain_copy[most_preferred_cell] = {value}
                # if most_preferred_cell == (7,0,4) and value == 14:
                #     print(domain_copy[most_preferred_cell])
                #     print(domain_copy[(7,3,4)])
                #     print("!!")
                # domain_copy = {key: value.copy() for (key, value) in self.domains.items()}
                children_node = Node(self.n, domain_copy, assigned_cell=most_preferred_cell)
                # if children_node.infer():
                self.children_nodes.append(children_node)

            self.has_expanded = True


    def select_unassigned_cell(self) -> (int, int):
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
        unassigned_cells = []
        for cell_key, cell_domains in self.domains.items():
            domain_size = len(cell_domains)
            if domain_size > 1:
                unassigned_cells.append(cell_key)
            # if domain_size < min_domain_size:
            #     min_domain_size = domain_size

        # Find the cells with the smallest domain size
        # min_domain_size = min(len(self.domains[cell]) for cell in unassigned_cells)
        min_domain_cells = []
        min_domain_size = self.n
        for cell in unassigned_cells:
            domain_size = len(self.domains[cell])
            if domain_size <= min_domain_size:
                min_domain_size = domain_size
                min_domain_cells.append(cell)

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

    def find_ordered_domain_values(self, cell_key) -> list[int]:
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
        domain_values = list(self.domains.get(cell_key))
        if len(domain_values) == 1:
            return domain_values

        def count_constraints(cell_key, value):
            count = 0
            for neighbor_key in self.find_cell_neighbours(cell_key):
                # If the value is in the domain of it's neighbours cells, then increment the count since this would now affect their domains
                if value in self.domains.get(neighbor_key):
                    count += 1
            return count

        domain_values.sort(key=lambda x: count_constraints(cell_key, x))

        return domain_values

    def find_arc(self, cell: (int, int, int)) -> {(int, int, int), (int, int, int)}:
        return self.all_arcs[cell]
    def find_degree(self, cell) -> int:
        degree = 0
        arcs = self.find_arc(cell)
        # print(arcs)
        for arc in arcs:
            other_cell = arc[1]
            if self.cell_is_empty(other_cell) == 0:
                degree += 1
        return degree

    def cell_is_empty(self, cell) -> bool:
        return len(self.domains[cell]) > 1

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
        # assignment = Assignments(board)
        # node = Node(assignment)
        self.n = len(board)

        domains_by_sub_square_index = []
        for i in range(self.n):
            domains_by_sub_square_index.append(set())

        domains = dict()
        for row in range(self.n):
            for col in range(self.n):
                sub_square_index = get_sub_square_index(self.n, row, col)
                cell_key = (row, col, get_sub_square_index(self.n, row, col))
                cell_value = board[row][col]
                if cell_value != 0:
                    domains[cell_key] = {cell_value}
                    domains_by_sub_square_index[sub_square_index].add(cell_value)
                else:
                    base_domains = set(range(1, self.n + 1))

                    for i in range(self.n):
                        base_domains.discard(board[row][i])
                        base_domains.discard(board[i][col])

                    for value in domains_by_sub_square_index[sub_square_index]:
                        base_domains.discard(value)

                    domains[cell_key] = base_domains if cell_value == 0 else {cell_value}

        self.stack = [Node(self.n, domains)]

    def solve123(self, max_process_seconds=None, mute=False):
        expiry_timestamp = (datetime.now() + timedelta(
            seconds=max_process_seconds)).timestamp() if max_process_seconds is not None else None

        i = 0
        timeout = 0
        stack_size = len(self.stack)
        while stack_size > 0:
            if (expiry_timestamp is not None) and (time.time() >= expiry_timestamp):
                raise Exception(f"No solution is found within {max_process_seconds} seconds")

            print("Stack length: ", len(self.stack))
            current_node = self.stack[-1]

            current_node.trim_domains()

            if current_node.has_cell_with_empty_domain():
                current_node.check()
                self.stack.pop()

            else:
                has_inferred = current_node.infer()
                if has_inferred:
                    print(current_node)

                    if current_node.is_solution():
                        return current_node

                    current_node.expand()
                    next_node = current_node.get_first_unchecked_child()

                    if next_node is None:
                        current_node.check()
                        self.stack.pop()
                    else:
                        self.stack.append(next_node)
                else:
                    current_node.check()
                    self.stack.pop()

            i += 1
            timeout += 1
            stack_size = len(self.stack)
            if timeout > 10000:
                timeout = 0
                for node in self.stack[1:]:
                    node.check()
                    self.stack.remove(node)
            if not mute and i % 1000 == 0:
                print()
                print(f"searching (i = {i}; timeout = {timeout}; stack size = {stack_size})")
                print(current_node)

    def solve(self):
        while len(self.stack) > 0:
            print("Stack length: ", len(self.stack))
            current_node = self.stack[-1]

            current_node.trim_domains()

            if current_node.has_cell_with_empty_domain():
                current_node.check()
                self.stack.pop()
            else:
                has_inferred = current_node.infer()
                if has_inferred:
                    print(current_node)

                    if current_node.is_solution():
                        return current_node

                    current_node.expand()
                    next_node = current_node.get_first_unchecked_child()

                    if next_node is None:
                        current_node.check()
                        self.stack.pop()
                    else:
                        self.stack.append(next_node)
                else:
                    current_node.check()
                    self.stack.pop()




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
    # puzzle = [[15, 8, 16, 14, 9, 13, 6, 1, 5, 3, 7, 4, 10, 2, 11, 12],
    #                         [6, 12, 11, 7, 5, 3, 10, 8, 1, 14, 9, 2, 13, 4, 16, 15],
    #                         [5, 13, 3, 10, 7, 16, 2, 4, 8, 11, 15, 12, 9, 6, 1, 14],
    #                         [9, 1, 2, 4, 11, 15, 14, 12, 16, 10, 13, 6, 8, 5, 7, 3],
    #                         [8, 7, 14, 13, 16, 5, 11, 10, 12, 15, 4, 1, 2, 3, 6, 9],
    #                         [16, 3, 10, 11, 13, 14, 4, 15, 7, 2, 6, 9, 5, 1, 12, 8],
    #                         [1, 9, 5, 15, 8, 6, 12, 2, 10, 13, 3, 11, 14, 16, 4, 7],
    #                         [2, 6, 4, 12, 3, 7, 1, 9, 14, 16, 5, 8, 15, 11, 13, 10],
    #                         [10, 4, 8, 16, 14, 12, 5, 7, 9, 1, 11, 13, 6, 15, 3, 2],
    #                         [11, 2, 9, 5, 15, 4, 16, 13, 6, 12, 8, 3, 7, 14, 10, 1],
    #                         [12, 15, 7, 6, 1, 10, 8, 3, 2, 4, 14, 5, 16, 13, 9, 11],
    #                         [13, 14, 1, 3, 2, 11, 9, 6, 15, 7, 16, 10, 12, 8, 5, 4],
    #                         [14, 5, 12, 1, 6, 8, 13, 11, 3, 9, 2, 7, 4, 10, 15, 16],
    #                         [3, 10, 6, 9, 4, 1, 15, 14, 13, 8, 12, 16, 11, 7, 2, 5],
    #                         [7, 11, 13, 8, 12, 2, 3, 16, 4, 5, 10, 15, 1, 9, 14, 6],
    #                         [4, 16, 15, 2, 10, 9, 7, 5, 11, 6, 1, 14, 3, 12, 8, 13]]
    # masked_board = mask_board(puzzle)

    masked_board = [[15, 0, 0, 0, 9, 13, 0, 0, 0, 3, 0, 4, 10, 0, 0, 0],
                    [6, 0, 11, 7, 0, 0, 10, 8, 1, 14, 0, 0, 13, 0, 0, 15],
                    [0, 13, 0, 0, 0, 16, 2, 0, 0, 11, 0, 12, 0, 0, 0, 0],
                    [0, 0, 0, 4, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 7, 0],
                    [0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
                    [0, 0, 10, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0],
                    [1, 0, 0, 15, 0, 0, 12, 0, 0, 0, 0, 0, 0, 16, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0],
                    [10, 4, 8, 0, 0, 0, 5, 0, 0, 1, 0, 0, 0, 15, 0, 0],
                    [11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0],
                    [0, 0, 7, 0, 1, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0],
                    [13, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 4],
                    [0, 5, 0, 0, 6, 0, 0, 0, 0, 9, 2, 0, 0, 0, 0, 0],
                    [0, 0, 0, 9, 0, 0, 0, 14, 13, 0, 0, 0, 0, 0, 2, 0],
                    [0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 1, 0, 0, 6], [0, 0, 15, 0, 0, 0, 0, 5, 0, 6, 0, 0, 3, 0, 0, 0]]

    solver = SudokuCspSolver(masked_board)
    result = solver.solve()
    print(result)


if __name__ == '__main__':
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(end_time - start_time)
