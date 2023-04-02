import multiprocessing
import os
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Tuple
from utils.benchmark_test.solved_board import get_solved_board
from sudoku_solver_brute_force import mask_board
from algo_util import get_sub_square_index
from log_util import logger
from slack_alert import AlertSender
import json
from uuid import uuid4
import ast


def load_node_from_json(node_id):
    with open(f"nodes/{node_id}.json", "r") as f:
        node_json = json.load(f)
        domains = {ast.literal_eval(key): set(value) for key, value in node_json["domains"].items()}
        node = Node(domains)
        node.assigned_cell = None if node_json["assigned_cell"] is None else tuple(node_json["assigned_cell"])
        node.cell_filled = node_json["cell_filled"]
        node.children = node_json["children"]
        node.id = node_json["id"]
        node.is_checked = node_json["is_checked"]
        node.is_expanded = node_json["is_expanded"]
        node.is_reserved = node_json["is_reserved"]
        return node

class SolverExecutionExpiredException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class PuzzleUnsolvedException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class InvalidAssignmentException(Exception):
    pass


class NeighborType(Enum):
    ROW = 1
    COL = 2
    SUB_SQUARE = 3


def solve_child(index, child_node_id, max_process_seconds=None):
    print(f"[DEBUG] Index = {index}; Solving child node: ", child_node_id)
    solver = SudokuSolverCsp()  # Initialize an empty SudokuSolverCsp
    solver.node_id_stack = [child_node_id]  # Set the stack to contain the single child
    return solver.solve_sequential(max_process_seconds)


class NodeEncoder(json.JSONEncoder):
    def default(self, node):
        if isinstance(node, Node):
            return {
                "assigned_cell": node.assigned_cell,
                "cell_filled": node.cell_filled,
                "children": list([child_node for child_node in node.children]),
                "domains": {str(key): list(value) for key, value in node.domains.items()},
                "id": node.id,
                "is_checked": node.is_checked,
                "is_expanded": node.is_expanded,
                "is_reserved": node.is_reserved,
            }
        return super().default(node)


class SudokuSolverCsp:
    def __init__(self, board: List[List[int]] = None) -> None:
        self.alert_sender = AlertSender()
        self.reserved_node_id_stack = []

        if board is None:
            # self.node_id_stack = [root]
            # self.reserved_node_id_stack = []
            return

        Node.n = len(board)
        domains = dict()
        for row in range(Node.n):
            for col in range(Node.n):
                key = (row, col, get_sub_square_index(Node.n, row, col))
                if board[row][col] != 0:
                    domains[key] = {board[row][col]}
                else:
                    domains[key] = set(range(1, Node.n + 1))

        first_node = Node(domains)
        self.node_id_stack = [first_node.id]


    def migrate_nodes_to_reserved_stack(self):
        for node_id in self.node_id_stack[1:]:
            node = load_node_from_json(node_id)
            node.reserve()
            self.reserved_node_id_stack.append(node.id)

        self.node_id_stack = self.node_id_stack[:1]

    def migrate_nodes_in_reserved_stack(self):
        self.node_id_stack = [Node.mark_node_as_unreserved(node_id) for node_id in self.reserved_node_id_stack]
        self.reserved_node_id_stack = []

    def solve(self, max_process_seconds=None, parallel=False):
        if parallel:
            return self.solve_parallel(max_process_seconds)
        else:
            return self.solve_sequential(max_process_seconds)

    def solve_parallel(self, max_process_seconds):
        # Generate all possible child nodes from the root node
        root_node_id = self.node_id_stack[0]
        root_node = load_node_from_json(root_node_id)
        root_node.expand()
        children_node_id = root_node.children

        # Set up a shared variable to store the first solution found
        first_solution = multiprocessing.Manager().Value("i", None)

        # Define a callback function to handle results from child processes
        def handle_result(result):
            if result is not None and first_solution.value is None:
                first_solution.value = result

        # Initialize the process pool with the number of available processors
        with multiprocessing.Pool() as pool:
            # Run the solve_child function for each child node in parallel
            for i in range(len(children_node_id)):
                child_node_id = children_node_id[i]
                pool.apply_async(solve_child, args=(i, child_node_id), callback=handle_result)

            while first_solution.value is None:
                pass
            return first_solution.value

    def solve_sequential(self, max_process_seconds):
        expiry_timestamp = (datetime.now() + timedelta(
            seconds=max_process_seconds)).timestamp() if max_process_seconds is not None else None
        i = 0
        timeout = 0

        # node = load_node_from_json(self.node_id_stack[0])
        # root = Node(node.domains)
        # self.node_id_stack = [root.id]
        print(self.node_id_stack)

        while self.node_id_stack:
            if (expiry_timestamp is not None) and (time.time() >= expiry_timestamp):
                raise SolverExecutionExpiredException(f"No solution is found within {max_process_seconds} seconds")

            current_node_id = self.node_id_stack[-1]
            current_node = load_node_from_json(current_node_id)

            if i % 5 == 0:
                msg = f"Iteration: #{i + 1}\npid: {os.getpid()}\nCell filled: {current_node.cell_filled}\nStack size: {len(self.node_id_stack)}"
                self.alert_sender.send(msg + "\n\n")
                if i % 20 == 0:
                    logger.info(msg)
                    logger.info(current_node)

            is_valid = current_node.do_forward_checking()

            if not is_valid:
                current_node.check()
                self.node_id_stack.pop()
                continue

            if current_node.is_solution():
                return current_node

            current_node.expand()
            next_node_id = current_node.get_first_traversable_child()

            if next_node_id is None:
                current_node.check()
                self.node_id_stack.pop()
            else:
                self.node_id_stack.append(next_node_id)

            i += 1
            timeout += 1
            if timeout >= 5000:
                timeout = 0
                self.migrate_nodes_to_reserved_stack()

            if len(self.node_id_stack) == 0:
                self.migrate_nodes_in_reserved_stack()


class Node:
    all_arcs = None
    every_cell_neighbour = None
    every_cell_neighbour_by_type = None
    n = None

    def __init__(self, domains,
                 assigned_cell=None,
                 new_value=None,
                 cell_filled=0
                 ) -> None:
        self.domains = {key: value.copy() for key, value in domains.items()}
        self.assigned_cell = assigned_cell
        if new_value is not None:
            self.domains[assigned_cell] = {new_value}
        self.children = []
        self.is_checked = False
        self.is_expanded = False
        self.is_reserved = False
        self.cell_filled = cell_filled
        # get unix epoch
        self.id = f"{int(time.time())}_{uuid4()}"

        if Node.all_arcs is None:
            Node.all_arcs = self.find_all_arcs()
        if Node.every_cell_neighbour is None:
            Node.every_cell_neighbour = self.find_every_cell_neighbours()
        if Node.every_cell_neighbour_by_type is None:
            Node.every_cell_neighbour_by_type = self.find_every_cell_neighbours_by_type()

        self.save()

    def save(self):
        json_format = json.dumps(self, cls=NodeEncoder, indent=4)
        with open(f"nodes/{self.id}.json", "w") as f:
            f.write(json_format)

    @classmethod
    def reset(cls):
        cls.all_arcs = None
        cls.every_cell_neighbour = None
        cls.every_cell_neighbour_by_type = None
        cls.n = None
        
    @staticmethod
    def get_arcs(cell: (int, int, int)) -> {(int, int, int), (int, int, int)}:
        return Node.all_arcs[cell]

    def find_all_arcs(self):
        all_arcs = dict()
        for cell in self.domains.keys():
            cell_arcs = set()
            row_index, col_index, sub_square_index = cell
            for counter_cell in self.domains.keys():
                counter_cell_row_index, counter_cell_col_index, counter_cell_sub_square_index = counter_cell
                is_same_row = row_index == counter_cell_row_index
                is_same_col = col_index == counter_cell_col_index
                is_same_sub_square = sub_square_index == counter_cell_sub_square_index
                is_different_cell = cell != counter_cell

                if is_different_cell and (is_same_row or is_same_col or is_same_sub_square):
                    cell_arcs.add((cell, counter_cell))
            all_arcs[cell] = cell_arcs

        return all_arcs

    def find_every_cell_neighbours(self):
        return {cell: self.find_cell_neighbours(cell) for cell in self.domains.keys()}

    def find_cell_neighbours(self, cell):
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

    def do_forward_checking(self):
        result = self.infer()
        # if result:
        #     self.apply_hidden_single_rule()
        # self.apply_naked_pair_rule()
        self.save()
        return result

    @staticmethod
    def mark_node_as_unreserved(node_id):
        node = load_node_from_json(node_id)
        node.unreserve()
        return node_id

    def reserve(self):
        self.is_reserved = True
        self.save()

    def unreserve(self):
        self.is_reserved = False
        self.save()

    def revise(self, cell_to_revise, cell_to_check):
        has_revised = False
        domain_of_cell_to_check: set = self.domains[cell_to_check]
        if len(domain_of_cell_to_check) == 1:
            # get one element from the set
            domain_value = next(iter(domain_of_cell_to_check))
            if domain_value in self.domains[cell_to_revise]:
                has_revised = True
                self.domains[cell_to_revise].remove(domain_value)
        return has_revised

    def infer(self):
        stack = []
        set_for_duplication_check = set()

        if self.assigned_cell is None:
            for _, arc_set in Node.all_arcs.items():
                for arc in arc_set:
                    stack.append(arc)
                    set_for_duplication_check.add(arc)
        else:
            for arc in self.get_arcs(self.assigned_cell):
                stack.append(arc)
                set_for_duplication_check.add(arc)

        while len(stack) > 0:
            arc_selected = stack.pop()
            set_for_duplication_check.remove(arc_selected)
            current_cell, other_cell = arc_selected
            has_revised = self.revise(cell_to_revise=other_cell, cell_to_check=current_cell)
            if has_revised:
                if len(self.domains[other_cell]) == 0:
                    return False
                else:
                    other_cell_neighbors = Node.every_cell_neighbour[other_cell]
                    for neighbor in other_cell_neighbors:
                        if neighbor != current_cell:
                            arc_to_prioritize = (other_cell, neighbor)
                            if arc_to_prioritize not in set_for_duplication_check:
                                stack.append(arc_to_prioritize)
                                set_for_duplication_check.add(arc_to_prioritize)
        return True

    def apply_naked_pair_rule(self):
        # Find if there is a pair of cells that have the same two values
        # If so, delete those values from the domain of the other cells in the same sub-square, row, and column

        if self.assigned_cell is None:
            return

        for key in Node.every_cell_neighbour[self.assigned_cell]:
            value = self.domains[key]
            for neighbor_key in Node.every_cell_neighbour[key]:
                neighbor_with_same_two_values = 0
                if self.domains[neighbor_key] == value:
                    neighbor_with_same_two_values += 1
                pair_found = neighbor_with_same_two_values == 2
                if pair_found:
                    for other_key in Node.every_cell_neighbour[key]:
                        if other_key != neighbor_key:
                            self.domains[other_key].discard(value[0])
                            self.domains[other_key].discard(value[1])

    def find_cell_neighbours_by_type(self, cell: (int, int, int), neighbor_type: NeighborType):
        cell_neighbours = set()
        row_index, col_index, sub_square_index = cell

        for counter_cell in self.domains.keys():
            counter_cell_row_index, counter_cell_col_index, counter_cell_sub_square_index = counter_cell
            is_different_cell = cell != counter_cell

            if neighbor_type == NeighborType.ROW:
                is_same_group = row_index == counter_cell_row_index
            elif neighbor_type == NeighborType.COL:
                is_same_group = col_index == counter_cell_col_index
            elif neighbor_type == NeighborType.SUB_SQUARE:
                is_same_group = sub_square_index == counter_cell_sub_square_index
            else:
                raise ValueError(f"Invalid neighbor_type: {neighbor_type}")

            if is_different_cell and is_same_group:
                cell_neighbours.add(counter_cell)

        return cell_neighbours

    def find_every_cell_neighbours_by_type(self):
        result = dict()
        for cell in self.domains.keys():
            row_neighbours = self.find_cell_neighbours_by_type(cell, NeighborType.ROW)
            col_neighbours = self.find_cell_neighbours_by_type(cell, NeighborType.COL)
            sub_square_neighbours = self.find_cell_neighbours_by_type(cell, NeighborType.SUB_SQUARE)
            result[cell] = {
                NeighborType.ROW: row_neighbours,
                NeighborType.COL: col_neighbours,
                NeighborType.SUB_SQUARE: sub_square_neighbours
            }
        return result

    def get_cell_neighbours_by_type(self, cell: (int, int, int), neighbor_type: NeighborType):
        return Node.every_cell_neighbour_by_type[cell][neighbor_type]

    def apply_hidden_single_rule(self):
        # Implement the "hidden single" inference rule
        # If a region contains only one square which can hold a specific number, then that number must go into that square
        if self.assigned_cell is None:
            return
        # Iterate through every unassigned cell in the grid
        for cell_key in Node.every_cell_neighbour[self.assigned_cell]:
            domain_values = self.domains[cell_key]
            if len(domain_values) == 1:
                continue

            row_neighbours = self.get_cell_neighbours_by_type(cell_key, NeighborType.ROW)
            col_neighbours = self.get_cell_neighbours_by_type(cell_key, NeighborType.COL)
            sub_square_neighbours = self.get_cell_neighbours_by_type(cell_key, NeighborType.SUB_SQUARE)

            def union_domain_values_of_neighbours(neighbours):
                union_domain_values = set()
                for neighbour in neighbours:
                    union_domain_values |= self.domains[neighbour]
                return union_domain_values

            union_domain_values_of_row_neighbours = union_domain_values_of_neighbours(row_neighbours)
            union_domain_values_of_col_neighbours = union_domain_values_of_neighbours(col_neighbours)
            union_domain_values_of_sub_square_neighbours = union_domain_values_of_neighbours(sub_square_neighbours)

            for domain_value in domain_values:
                if (
                        domain_value not in union_domain_values_of_row_neighbours
                        and domain_value not in union_domain_values_of_col_neighbours
                        and domain_value not in union_domain_values_of_sub_square_neighbours
                ):
                    self.domains[cell_key] = {domain_value}
                    break

    def expand(self):
        if not self.is_expanded:
            # constraint_domain_copy = {key: value for key, value in self.constraints.domains.items()}

            cell_selected = self.select_unassigned_cell()

            for value in self.find_ordered_domain_values(cell_selected):
                new_node = Node(self.domains, cell_selected, value, cell_filled=self.cell_filled + 1)
                self.children.append(new_node.id)

            self.is_expanded = True
            self.save()

    def cell_is_empty(self, cell: Tuple[int, int]):
        return len(self.domains[cell]) > 1

    def select_unassigned_cell(self) -> Tuple[int, int]:
        unassigned_cells = [cell for cell in self.domains.keys() if self.cell_is_empty(cell)]

        # Find the cells with the smallest domain size
        min_domain_size = min(len(self.domains[cell]) for cell in unassigned_cells)
        min_domain_cells = [cell for cell in unassigned_cells if len(self.domains[cell]) == min_domain_size]

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

    def find_degree(self, cell) -> int:
        degree = 0
        arcs = self.get_arcs(cell)
        for arc in arcs:
            other_cell = arc[1]
            if self.cell_is_empty(other_cell) == 0:
                degree += 1
        return degree

    def find_ordered_domain_values(self, cell_key):
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
        domain_values = list(self.domains[cell_key])
        if len(domain_values) == 1:
            return domain_values

        domain_values.sort(key=lambda x: self.count_constraints(cell_key, x))

        return domain_values

    def count_constraints(self, cell_key, value):
        count = 0
        for neighbor_key in self.every_cell_neighbour[cell_key]:
            # If the value is in the domain of it's neighbours cells, then increment the count since this would now affect their domains
            if value in self.domains[neighbor_key]:
                count += 1
        return count

    def check(self):
        self.is_checked = True
        self.save()

    def __str__(self):
        sub_square_size = (int(self.n ** 0.5), int(self.n ** 0.5))

        if self.n == 12:
            sub_square_size = (3, 4)

        full_row = "+".join(["-" * (sub_square_size[1] * 5 - 1)] * sub_square_size[0])

        board_str = ''
        for row in range(self.n):
            if row % sub_square_size[0] == 0:
                board_str += full_row + '\n'
            row_str = ' |'
            for col in range(self.n):
                domain = self.domains[(row, col, get_sub_square_index(Node.n, row, col))]
                if len(domain) == 1:
                    value = next(iter(domain))
                else:
                    value = 0
                if value == 0:
                    row_str += '__'
                else:
                    row_str += f'{value} ' if value < 10 else f'{value}'
                if (col + 1) % sub_square_size[1] == 0:
                    row_str += '  |'
                row_str += "  "
            board_str += row_str + '\n'
        board_str += full_row

        return board_str + '\n'

    def is_solution(self):
        """
        Check if this set of assignments is a solution to the problem (the whole board is filled and satisfies the constraints)
        """
        for domain_values in self.domains.values():
            if len(domain_values) != 1:
                return False
        all_arcs = set.union(*self.all_arcs.values())
        for arc in all_arcs:
            cell_one, cell_two = arc
            cell_one_value = next(iter(self.domains[cell_one]))
            cell_two_value = next(iter(self.domains[cell_two]))
            if cell_one_value == cell_two_value:
                raise InvalidAssignmentException(
                    f"{cell_one} and {cell_two} were both assigned the same value of {cell_one_value}\n{self}")
        return True

    def get_first_traversable_child(self):
        nodes = [load_node_from_json(child_node) for child_node in self.children]
        for node in nodes:
            if not node.is_checked and (node.is_reserved is False):
                return node.id
        return None

    def to_2d_array(self):
        two_d_array = []

        for i in range(Node.n):
            arr = list()
            for j in range(Node.n):
                arr.append(0)
            two_d_array.append(arr)

        for key, value in self.domains.items():
            row_index, col_index, _ = key
            two_d_array[row_index][col_index] = value

        return two_d_array


def solve_with_csp_iterative(board):
    Node.reset()
    solver = SudokuSolverCsp(board)
    if Node.n <= 12:
        result = solver.solve(parallel=False)
    else:
        result = solver.solve(parallel=True)
    return result.to_2d_array()


def main():
    board = get_solved_board(12)
    masked_board = mask_board(board)
    print(masked_board)
    start_time = time.time()
    sudoku_solver = SudokuSolverCsp(masked_board)
    result = sudoku_solver.solve(parallel=True)
    end_time = time.time()
    print("Solution")
    print(result)
    print(f"Solved in {end_time - start_time} seconds")


if __name__ == '__main__':
    main()
