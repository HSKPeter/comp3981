from enum import Enum, auto
import abc
from sudoku_solver_csp_recursive import Assignments, Constraints, backtrack
from sudoku_solver_brute_force import SudokuSolver as SudokuSolverBruteForce
from sudoku_solver_csp_iterative import SudokuSolverCsp as SudokuSolverIterative
from csp_azure import SudokuSolverCsp as SudokuSolverAzure


class AlgorithmType(Enum):
    BRUTE_FORCE = auto()
    CSP_RECURSIVE = auto()
    CSP_ITERATIVE = auto()
    CSP_ITERATIVE_MULTIPROCESS = auto()
    CSP_AZURE = auto()


class AlgorithmRunner(abc.ABC):
    @abc.abstractmethod
    def solve_sudoku(self, board):
        pass


class BruteForceAlgorithmRunner(AlgorithmRunner):
    def solve_sudoku(self, board):
        sudoku_solver = SudokuSolverBruteForce(board)
        return sudoku_solver.solve()


class RecursiveCspAlgorithmRunner(AlgorithmRunner):
    def solve_sudoku(self, board):
        assignments = Assignments(board)
        constraints = Constraints(assignments)
        return backtrack(constraints, assignments, mute=True)


class IterativeCspAlgorithmRunner(AlgorithmRunner):

    def __init__(self, is_parallel=False):
        self.is_parallel = is_parallel

    def solve_sudoku(self, board):
        sudoku_solver = SudokuSolverIterative(board)
        return sudoku_solver.solve(parallel=self.is_parallel)


class AzureCspAlgorithmRunner(AlgorithmRunner):

    def __init__(self, is_parallel=False):
        self.is_parallel = is_parallel

    def solve_sudoku(self, board):
        sudoku_solver = SudokuSolverAzure(board)
        return sudoku_solver.solve(parallel=self.is_parallel)
