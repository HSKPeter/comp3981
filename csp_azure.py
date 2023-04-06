"""
This module houses the CSP solver customized for execution on Azure VM instance.
Customizations are made to improve logging and monitoring, considering the execution would take a longer time.
Also, Azure storage is used to externalize the memory to avoid out-of-memory issue.
"""

import multiprocessing
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Tuple, Optional, Dict
from algo_util import get_sub_square_index
from slack_alert import AlertSender
from sudoku_solver_brute_force import mask_board
from log_util import logger, log_format
import threading
import os
import ast
from uuid import uuid4
from azure.storage.blob import BlobServiceClient
import json
import os


class AzureStorageClient:
    _alert_sender = AlertSender()

    def __init__(self, container_name):
        """
        Initialize an instance of AzureStorageClient
        """
        self.container_name = container_name

        connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        self.connection_string = connection_string if connection_string is not None else input("Enter Azure Storage Connection String: ")

        if self.connection_string is None:
            raise Exception("No connection string provided")

        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.container_client = None

        if self.check_container_exists():
            self.container_client = self.blob_service_client.get_container_client(self.container_name)
        else:
            self.create_container()
            self.container_client = self.blob_service_client.get_container_client(self.container_name)

        self._instance = self

    def check_container_exists(self):
        """
        Check if a container exists in Azure Blob Storage
        """
        return self.blob_service_client.get_container_client(self.container_name).exists()

    def create_container(self):
        """
        Create a container in Azure Blob Storage
        """
        self.blob_service_client.create_container(self.container_name)
        logger.info(f"Created Azure Storage container {self.container_name}")
        self._alert_sender.send(f"Created Azure Storage container {self.container_name}")

    def upload_file(self, file_name, data):
        """
        Upload a json string to Azure Blob Storage as json file
        """
        blob_client = self.container_client.get_blob_client(file_name)
        blob_client.upload_blob(json.dumps(data), overwrite=True)

    def download_data(self, file_name):
        """
        Download a json file from Azure Blob Storage
        """
        blob_client = self.container_client.get_blob_client(file_name)
        data = blob_client.download_blob().readall()
        json_str = data.decode('utf-8')
        return json.loads(json_str)


def load_node_from_json(storage_client, node_id):
    node_data = storage_client.download_data(f"{node_id}.json")
    domains = {ast.literal_eval(key): set(value) for key, value in node_data["domains"].items()}
    node = Node(domains, storage_client)
    node.assigned_cell = None if node_data["assigned_cell"] is None else tuple(node_data["assigned_cell"])
    node.children = node_data["children"]
    node.id = node_data["id"]
    node.is_checked = node_data["is_checked"]
    node.is_expanded = node_data["is_expanded"]
    node.is_reserved = node_data["is_reserved"]
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


class SudokuSolverCsp:
    """
    A class that solves a Sudoku board using a CSP approach.

    This class can be used to solve a Sudoku board using parallel processing. This is done by creating a SudokuSolverCsp
    for each child of the root node, and solving each of them in parallel. The first child to find a solution will be
    returned, and the other children will be discarded.

    Attributes:
        node_id_stack: A list of Node objects representing the search tree
        reserved_node_id_stack: A list of Node objects representing the search tree. Used if the solution is not found after a
            certain number of nodes have been visited.
    """

    def __init__(self, board: List[List[int]] = None, root: str = None) -> None:
        """
        Creates a SudokuSolverCsp object from a board or a root node.

        Args:
            board: A 2D list of integers representing a Sudoku board, can be None if root is not None
            root: A Node object representing the root node of the search tree, can be None if board is not None
        """
        self.alert_sender = AlertSender()
        self.container_name = str(int(time.time())) + uuid4().hex
        self.storage_client = AzureStorageClient(self.container_name)

        # Creating a SudokuSolverCsp object from a board. Used for solving subtrees in parallel.
        if board is None:
            self.node_id_stack = [root]
            self.reserved_node_id_stack = []
            return

        Node.reset()
        Node.n = len(board)
        domains = dict()
        for row in range(Node.n):
            for col in range(Node.n):
                key = (row, col, get_sub_square_index(Node.n, row, col))
                if board[row][col] != 0:
                    domains[key] = {board[row][col]}
                else:
                    domains[key] = set(range(1, Node.n + 1))

        first_node = Node(domains, self.storage_client)
        self.node_id_stack = [first_node.id]
        self.reserved_node_id_stack = []

    def migrate_nodes_to_reserved_stack(self):
        """
        Migrates all nodes in the stack to the reserved stack, except for the root node.

        This is used if the solution is not found after a certain number of nodes have been visited. This allows us to
        search in a different subtree.
        """
        for node_id in self.node_id_stack[1:]:
            node = load_node_from_json(self.storage_client, node_id)
            node.reserve()
            self.reserved_node_id_stack.append(node_id)

        self.node_id_stack = self.node_id_stack[:1]

    def migrate_nodes_in_reserved_stack(self):
        """
        Migrates all nodes in the reserved stack to the stack.

        This is used if all the children of the root node have been visited and all the subtrees have been moved to the
        reserved stack. This allows us to search in the subtrees again.
        """

        def mark_node_as_unreserved(node_id):
            node = load_node_from_json(self.storage_client, node_id)
            node.unreserve()
            return node_id

        self.node_id_stack = [mark_node_as_unreserved(node_id) for node_id in self.reserved_node_id_stack]
        self.reserved_node_id_stack = []

    def solve(self, max_process_seconds: int = None, parallel: bool = False):
        """
        Solves the Sudoku puzzle using a backtracking algorithm.
        Args:
            max_process_seconds: The maximum number of seconds to run the solver for.
            parallel: Whether to use a parallelized version of the backtracking algorithm.

        Returns:
            A Node object representing the solution to the Sudoku puzzle. If no solution is found, returns None.
        """
        logger.info(f"parallel {parallel}")
        if parallel:
            return self.solve_parallel(max_process_seconds)
        else:
            return self.solve_sequential(max_process_seconds)

    @staticmethod
    def solve_child(child_node_id: str, max_process_seconds: int = None):
        """
        Helper method for solving a child node. Used for parallelizing the backtracking algorithm.

        Args:
            child_node: The child node to solve. This will be the root node of the subtree.
            max_process_seconds: The maximum number of seconds to run the solver for.

        Returns:
            A Node object representing the solution to the Sudoku puzzle. If no solution is found, returns None.
        """
        solver = SudokuSolverCsp()  # Initialize an empty SudokuSolverCsp
        solver.node_id_stack = [child_node_id]  # Set the stack to contain the single child
        return solver.solve_sequential(max_process_seconds)

    def solve_parallel(self, max_process_seconds):
        """
        Solves the Sudoku puzzle using a parallelized version of the backtracking algorithm.

        The parallelization is done by using a process pool to run the backtracking algorithm on each child node of
        the root node in parallel.

        Args:
            max_process_seconds: The maximum number of seconds to run the solver for.

        Returns:
            A Node object representing the solution to the Sudoku puzzle. If no solution is found, returns None.
        """
        # Generate all possible child nodes from the root node
        root_node_id = self.node_id_stack[0]
        root_node = load_node_from_json(self.storage_client, root_node_id)
        root_node.expand()
        children_ids = root_node.children

        # Set up a shared variable to store the first solution found
        first_solution = multiprocessing.Manager().Value("i", None)

        # Define a callback function to update the shared variable when the first solution is found
        def handle_result(result):
            if result is not None and first_solution.value is None:
                first_solution.value = result

        # Initialize the process pool with the number of available processors
        with multiprocessing.Pool() as pool:

            # Run the solve_child function for each child node in parallel
            for child_id in children_ids:
                pool.apply_async(self.solve_child, args=(child_id,), callback=handle_result)

            # Wait until the first solution is found
            while first_solution.value is None:
                pass

            return first_solution.value
        
    def solve_sequential(self, max_process_seconds):
        """
        Solves the Sudoku puzzle using the backtracking CSP algorithm.

        1. Creates the root node from the first domain in the stack and initializes the stack with the root node.
        2. Iteratively processes nodes in the stack:
            a. If the expiry timestamp is reached, raises a SolverExecutionExpiredException.
            b. Performs forward checking on the current node.
            c. If the node is not valid, backtracks to the previous node in the stack.
            d. If the node is a solution, returns the node.
            e. If the node has children, expands the node and moves to the next child.
            f. Updates a timeout counter to periodically migrate nodes between stacks.
        4. If the stack is empty, migrates nodes from the reserved stack and continues processing.

        Args:
            max_process_seconds: The maximum number of seconds to run the solver for.

        Returns:
            A Node object representing the solution. If no solution is found, returns None.
        """
        expiry_timestamp = (datetime.now() + timedelta(
            seconds=max_process_seconds)).timestamp() if max_process_seconds is not None else None
        i = 0
        timeout = 0

        root_node_id = self.node_id_stack[0]
        self.node_id_stack = [root_node_id]

        while self.node_id_stack:
            if (expiry_timestamp is not None) and (time.time() >= expiry_timestamp):
                raise SolverExecutionExpiredException(f"No solution is found within {max_process_seconds} seconds")

            current_node_id = self.node_id_stack[-1]
            current_node = load_node_from_json(self.storage_client, current_node_id)

            if i % 10 == 0:
                msg = f"Iteration: #{i + 1}\nThread ID: {threading.get_ident()}\npid: {os.getpid()}\nNode ID: {current_node.id}\nStack size: {len(self.node_id_stack)}"
                self.alert_sender.send(msg + "\n\n")
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
            next_node = current_node.get_first_traversable_child()

            if next_node is None:
                current_node.check()
                self.node_id_stack.pop()
            else:
                self.node_id_stack.append(next_node.id)

            i += 1
            timeout += 1
            if timeout >= 5000:
                timeout = 0
                self.migrate_nodes_to_reserved_stack()

            if len(self.node_id_stack) == 0:
                self.migrate_nodes_in_reserved_stack()


class Node:
    """
    A node in the search tree.

    Attributes:
        domains: A dictionary of domains for each cell in the Sudoku puzzle.
        assigned_cell: The cell that was assigned a value in the previous node.
        children: A list of child nodes.
        is_checked: A boolean indicating whether the node has been checked.
        is_expanded: A boolean indicating whether the node has been expanded.
        is_reserved: A boolean indicating whether the node has been moved to the reserve stack.
    """
    # a dictionary of all arcs in the Sudoku puzzle. key: cell, value: list of arcs for cell
    all_arcs = None

    # a dictionary of all neighbours for each cell in the Sudoku puzzle. key: cell, value: list of neighbours for cell
    every_cell_neighbour = None

    # a dictionary of all neighbours seperated by type (row, column, box) for each cell in the Sudoku puzzle.
    # key: cell, value: dictionary of neighbours seperated by type
    every_cell_neighbour_by_type = None

    # length of the board
    n = None

    def __init__(self, domains, storage_client, assigned_cell=None, new_value=None) -> None:
        """
        Initializes a Node object.

        Args:
            domains: A dictionary of domains for each cell in the Sudoku puzzle. A domain of one means that the cell is
                assigned a value.
            assigned_cell: The cell that was assigned a value in the previous node.
            new_value: The value that was assigned to the assigned_cell in the previous node.
        """
        self.domains = {key: value.copy() for key, value in domains.items()}
        self.assigned_cell = assigned_cell
        if new_value is not None:
            self.domains[assigned_cell] = {new_value}
        self.children = []
        self.is_checked = False
        self.is_expanded = False
        self.is_reserved = False
        self.storage_client = storage_client
        self.id = f"{int(time.time())}_{uuid4()}"

        if Node.all_arcs is None:
            Node.all_arcs = self.find_all_arcs()
        if Node.every_cell_neighbour is None:
            Node.every_cell_neighbour = self.find_every_cell_neighbours()
        if Node.every_cell_neighbour_by_type is None:
            Node.every_cell_neighbour_by_type = self.find_every_cell_neighbours_by_type()

        self.save()

    @classmethod
    def reset(cls):
        """
        Resets the class variables.
        """
        cls.all_arcs = None
        cls.every_cell_neighbour = None
        cls.every_cell_neighbour_by_type = None
        cls.n = None

    @staticmethod
    def get_arcs(cell: Tuple[int, int, int]) -> set[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
        """
        Returns the arcs for a given cell. Each arch is a tuple of two cells.

        Args:
            cell: The cell to find the arcs for.

        Returns: a set of arcs for the given cell.
        """
        return Node.all_arcs[cell]

    def find_all_arcs(self):
        """
        Finds all arcs in the Sudoku puzzle.

        Returns:
            a dictionary of all arcs in the Sudoku puzzle. key: cell, value: list of arcs for cell
        """
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
        """
        Finds all neighbours for each cell in the Sudoku puzzle.

        Returns:
            a dictionary of all neighbours for each cell in the Sudoku puzzle. key: cell, value: list of neighbours for
            cell

        """
        return {cell: self.find_cell_neighbours(cell) for cell in self.domains.keys()}

    def find_cell_neighbours(self, cell: Tuple[int, int, int]) -> set[Tuple[int, int, int]]:
        """
        Finds all neighbours for a given cell.

        Args:
            cell: The cell to find the neighbours for.

        Returns:
            a list of neighbours for the given cell.
        """
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

    def do_forward_checking(self) -> bool:
        """
        Does forward checking on the domains of the cells in the Sudoku puzzle.

        Forward checking is done by applying the AC3 algorithm on the arcs of the Sudoku puzzle.
        Additional forward checking heuristics can also be applied here.

        Returns:
            True if the domains of the cells in the Sudoku puzzle are consistent, False otherwise.
        """
        result = self.infer()
        if result:
            self.apply_hidden_single_rule()
            self.apply_naked_pair_rule()
        return result

    def reserve(self):
        """
        Marks the node as reserved.
        """
        self.is_reserved = True
        self.save()

    def unreserve(self):
        """
        Marks the node as unreserved.
        """
        self.is_reserved = False
        self.save()

    def revise(self, cell_to_revise: Tuple[int, int, int], cell_to_check: Tuple[int, int, int]) -> bool:
        """
        Revise the domain of a cell based on the domain of another cell.

        The revision is done by removing a value of the domain of the cell to revise if it the only value remaining in
        the domain of the cell to check.

        Args:
            cell_to_revise: The cell to revise.
            cell_to_check: The cell to check.

        Returns:
            True if the domain of the cell to revise has been revised, False otherwise.
        """
        has_revised = False
        domain_of_cell_to_check: set = self.domains[cell_to_check]
        if len(domain_of_cell_to_check) == 1:
            # get one element from the set
            domain_value = next(iter(domain_of_cell_to_check))
            if domain_value in self.domains[cell_to_revise]:
                has_revised = True
                self.domains[cell_to_revise].remove(domain_value)
        return has_revised

    def infer(self) -> bool:
        """
        Infers the domains of the cells in the Sudoku puzzle.

        This function uses the Maintaining Arc Consistency (MAC) heuristic, which is based on the
        AC-3 algorithm. This helps ensure that the remaining variables maintain their arc consistency
        after assigning a value to the current variable.

        Returns:
            True if the domains of the cells in the Sudoku puzzle are consistent, False otherwise.
        """
        stack = []
        set_for_duplication_check = set()

        # if the node has no assignment (root node), then we need to check all arcs
        if self.assigned_cell is None:
            for _, arc_set in Node.all_arcs.items():
                for arc in arc_set:
                    stack.append(arc)
                    set_for_duplication_check.add(arc)
        else:
            # if the node has an assignment, then we only need to check the arcs of the assigned cell
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
        """
        Applies the naked pair rule to the domains of the cells in the Sudoku puzzle.

        Find if there is a pair of cells that have the same two values
        If so, delete those values from the domain of the other cells in the same sub-square, row, and column

        This function is currently not used as it was too computationally expensive.
        """
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

    def find_cell_neighbours_by_type(self, cell: Tuple[int, int, int], neighbor_type: NeighborType):
        """
        Finds the neighbours of a cell by type (row, column, or sub-square).
        Args:
            cell: The cell to find the neighbours of.
            neighbor_type: The type of neighbour to find.

        Returns:
            A set of cells that are neighbours of the given cell.
        """
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

    def find_every_cell_neighbours_by_type(self) -> \
            Dict[Tuple[int, int, int], Dict[NeighborType, set[Tuple[int, int, int]]]]:
        """
        Finds the neighbours of every cell by type (row, column, or sub-square).
        Returns:
            A dictionary of cells to a dictionary of neighbour types to a set of cells.
        """
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

    @staticmethod
    def get_cell_neighbours_by_type(cell: (int, int, int), neighbor_type: NeighborType):
        """
        Gets the neighbours of a cell by type (row, column, or sub-square).

        Args:
            cell: The cell to find the neighbours of.
            neighbor_type: The type of neighbour to find.
        Returns: A set of cells that are neighbours of the given cell.
        """
        return Node.every_cell_neighbour_by_type[cell][neighbor_type]

    def apply_hidden_single_rule(self):
        """
        Implement the "hidden single" inference rule
        If a region contains only one square which can hold a specific number, then that number must go into that square
        """

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
        """
        Expand the node by finding all possible values for the next unassigned cell and creating a child node for each
        possible value.

        """
        if not self.is_expanded:
            cell_selected = self.select_unassigned_cell()

            for value in self.find_ordered_domain_values(cell_selected):
                new_node = Node(self.domains, self.storage_client, cell_selected, value)
                self.children.append(new_node.id)

            self.is_expanded = True

            self.save()

    def cell_is_empty(self, cell: Tuple[int, int, int]):
        """
        Checks if a cell is empty.
        Args:
            cell: The cell to check.

        Returns: True if the cell is empty, False otherwise.
        """
        return len(self.domains[cell]) > 1

    def select_unassigned_cell(self) -> Tuple[int, int, int]:
        """
        Selecting unassigned variables: Use a combination of the Minimum Remaining Values (MRV) and Degree heuristics.

        MRV: Choose the variable with the fewest legal values remaining in its domain. Applied first.

        Degree: Choose the variable involved in the highest number of constraints with other unassigned variables.
        Applied second if there is a tie for MRV.

        Returns: The cell to be used in assignment.

        """
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

    def find_degree(self, cell: Tuple[int, int, int]) -> int:
        """
        Find the degree of a cell based on the number of arcs it has to empty cells.
        Args:
            cell: The cell to find the degree of.

        Returns: The degree of the cell as an integer.
        """
        degree = 0
        arcs = self.get_arcs(cell)
        for arc in arcs:
            other_cell = arc[1]
            if self.cell_is_empty(other_cell):
                degree += 1
        return degree

    def find_ordered_domain_values(self, cell_key: Tuple[int, int, int]):
        """
        Find the domain values of a cell in order of least constraining value first.

        Ordering values of a variable: Use the Least Constraining Value (LCV) heuristic, which selects the value that
        imposes the fewest constraints on the remaining variables. The Least Constraining Value (LCV) heuristic is used
        to order the values of a variable when attempting to assign a value during the search process in a Constraint
        Satisfaction Problem (CSP). The LCV heuristic aims to minimize the impact of the current assignment on the
        future assignments of other variables.

        Args:
            cell_key: The cell to find the domain values of.

        Returns: The domain values of the cell in order of least constraining value first.
        """
        domain_values = list(self.domains[cell_key])
        if len(domain_values) == 1:
            return domain_values

        domain_values.sort(key=lambda x: self.count_constraints(cell_key, x))

        return domain_values

    def count_constraints(self, cell_key: Tuple[int, int, int], value: int):
        """
        Count the number of constraints a value would impose on the domains of the neighbours of a cell.
        Args:
            cell_key: The cell to find the neighbours of.
            value: The value to count the constraints of.

        Returns: The number of constraints the value would impose on the domains of the neighbours of the cell.
        """
        count = 0
        for neighbor_key in self.every_cell_neighbour[cell_key]:
            # If the value is in the domain of it's neighbours cells, then increment the count since this would now
            # affect their domains
            if value in self.domains[neighbor_key]:
                count += 1
        return count

    def check(self):
        """
        Mark this node as checked. Checked nodes are exhausted and will not be visited again.
        """
        self.is_checked = True
        self.save()

    def __str__(self):
        """
        Print the board in a readable format.
        """
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

    def is_solution(self) -> bool:
        """
        Check if the board is a solution (the whole board is filled and satisfies the constraints).

        Throws:
            InvalidAssignmentException: If the board is filled but does not satisfy the constraints.
        Returns:
            True if the board is a solution, False otherwise.
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

    def get_first_traversable_child(self) -> Optional['Node']:
        """
        Get the first child of this node that is not checked and is not reserved.

        Returns:
            The first child of this node that is not checked and is not reserved.
        """
        for node_id in self.children:
            node = load_node_from_json(self.storage_client, node_id)
            if not node.is_checked and (node.is_reserved is False):
                return node
        return None

    def to_2d_array(self):
        """
        Convert the board to a 2d array.

        Returns:
            The board as a 2d array.
        """
        two_d_array = []

        for i in range(Node.n):
            arr = list()
            for j in range(Node.n):
                arr.append(0)
            two_d_array.append(arr)

        for key, value in self.domains.items():
            row_index, col_index, _ = key
            two_d_array[row_index][col_index] = next(iter(value))

        return two_d_array
    
    def save(self):
        self.storage_client.upload_file(f"{self.id}.json", self.to_dict())

    def to_dict(self):
        return {
            "assigned_cell": self.assigned_cell,
            "children": list([child_node for child_node in self.children]),
            "domains": {str(key): list(value) for key, value in self.domains.items()},
            "id": self.id,
            "is_checked": self.is_checked,
            "is_expanded": self.is_expanded,
            "is_reserved": self.is_reserved,
        }    


def solve_with_csp_iterative(board: List[List[int]]) -> List[List[int]]:
    """
    Solve a sudoku board using the CSP iterative algorithm.

    Args:
        board: The board to solve as a 2d array.

    Returns:
        The solved board as a 2d array.
    """
    solver = SudokuSolverCsp(board)
    result = solver.solve(parallel=False)
    # if Node.n <= 12:
    #     result = solver.solve(parallel=False)
    # else:
    #     result = solver.solve(parallel=True)
    return result.to_2d_array()


def main():
    """
    Main function to run the program. Used for testing purposes.
    """

    logger.remove()

    now = datetime.now()
    formatted_date_time = now.strftime('%Y-%m-%d_%H%M')

    package_directory = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(package_directory, "log", f"{formatted_date_time}_exploration_{uuid4().hex}.log")
    logger.add(log_file_path, format=log_format)

    # NOTE: Put the board puzzle here
    board = []

    logger.info(board)
    start_time = time.time()
    sudoku_solver = SudokuSolverCsp(board)
    result = sudoku_solver.solve(parallel=False)
    end_time = time.time()
    logger.info("Solution")
    logger.info(result)
    logger.info(f"Solved in {end_time - start_time} seconds")


if __name__ == '__main__':
    main()