from enum import Enum, auto
import abc
from sudoku_solver_csp import Assignments, Constraints, backtrack
from sudoku_solver import SudokuSolver
from iterative_refactored import SudokuSolverCsp as SudokuSolverIterative


class AlgorithmType(Enum):
    BRUTE_FORCE = auto()
    CSP_RECURSIVE = auto()
    CSP_ITERATIVE = auto()


class AlgorithmRunner(abc.ABC):
    @abc.abstractmethod
    def solve_sudoku(self, board):
        pass


class BruteForceAlgorithmRunner(AlgorithmRunner):
    def solve_sudoku(self, board):
        sudoku_solver = SudokuSolver(board)
        return sudoku_solver.solve()


class RecursiveCspAlgorithmRunner(AlgorithmRunner):
    def solve_sudoku(self, board):
        assignments = Assignments(board)
        constraints = Constraints(assignments)
        return backtrack(constraints, assignments, mute=True)


class IterativeCspAlgorithmRunner(AlgorithmRunner):
    def solve_sudoku(self, board):
        sudoku_solver = SudokuSolverIterative(board)
        return sudoku_solver.solve()