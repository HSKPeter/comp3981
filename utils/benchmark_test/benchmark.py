import time
from datetime import datetime
from puzzle_loader import PuzzleLoader
from sudoku_solver_csp import Assignments, Constraints, backtrack
from sudoku_solver import mask_board, SudokuSolver
from iterative_refactored import SudokuSolverCsp as SudokuSolverIterative
from enum import Enum, auto
import abc


class PuzzleGenerator:
    @staticmethod
    def read_file(filename):
        with open(filename, 'r') as file:
            return file.read()


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


class BenchmarkTestRunner:
    def __init__(self, board_size=9, num_runs=10):
        self._board_size = board_size
        self._num_runs = num_runs
        self._algorithm_runners = None
        self._algo_type_name = None

    def set_algorithm_runners(self, algo_type):
        self._algo_type_name = algo_type.name.lower()

        if algo_type == Algorithm.CSP_RECURSIVE:
            self._algorithm_runners = RecursiveCspAlgorithmRunner()
        elif algo_type == Algorithm.CSP_ITERATIVE:
            self._algorithm_runners = IterativeCspAlgorithmRunner()
        elif algo_type == Algorithm.BRUTE_FORCE:
            self._algorithm_runners = BruteForceAlgorithmRunner()

    def run_benchmark(self, board):
        start_time = time.time()
        result = self._algorithm_runners.solve_sudoku(board)
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Solution found in {elapsed_time}: \n{result}\n")

    def test_performance_by_iteration(self, board, description=None):
        if description is None:
            description = f"{self._board_size}x{self._board_size} benchmark"
        total_elapsed_time = 0
        failure_count = 0
        total_failure_time = 0

        for i in range(self._num_runs):
            start_time = time.time()
            result = self._algorithm_runners.solve_sudoku(board)
            end_time = time.time()
            elapsed_time = end_time - start_time

            print(f"Test {i + 1} result\n{result}\n")

            if result is None:
                failure_count += 1
                total_failure_time += elapsed_time
            else:
                total_elapsed_time += elapsed_time

        successful_runs = self._num_runs - failure_count
        average_elapsed_time = total_elapsed_time / successful_runs if successful_runs > 0 else 0
        average_failure_time = total_failure_time / failure_count if failure_count > 0 else 0
        result = f"{self._algo_type_name}\ndescription: {description}\n" \
                 f"average runtime (seconds) over {successful_runs} successful runs: {average_elapsed_time:.6f}\n" \
                 f"failures: {failure_count} (average time for failures: {average_failure_time:.6f})\n\n"

        now = datetime.now()
        formatted_date_time = now.strftime('%Y-%m-%d_%H%M')
        with open(f"{formatted_date_time}_{self._algo_type_name}.txt", 'a') as file:
            file.write(result)


class Algorithm(Enum):
    BRUTE_FORCE = auto()
    CSP_RECURSIVE = auto()
    CSP_ITERATIVE = auto()


def main():
    benchmark_test_runner = BenchmarkTestRunner()
    puzzle_loader = PuzzleLoader()

    file_path, raw_content, board = puzzle_loader.load_unsolved_9x9_puzzle_from_standard_samples(is_easy=True)

    print(f"Loaded puzzle from {file_path}:\n{raw_content}\n")

    benchmark_test_runner.set_algorithm_runners(Algorithm.BRUTE_FORCE)
    benchmark_test_runner.run_benchmark(board)

    benchmark_test_runner.set_algorithm_runners(Algorithm.CSP_RECURSIVE)
    benchmark_test_runner.run_benchmark(board)

    benchmark_test_runner.set_algorithm_runners(Algorithm.CSP_ITERATIVE)
    benchmark_test_runner.run_benchmark(board)


if __name__ == '__main__':
    main()
