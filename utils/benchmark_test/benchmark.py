import time
from solved_board import get_solved_board
from datetime import datetime
from sudoku_solver_csp_recursive import Assignments, Constraints, backtrack
from sudoku_solver_brute_force import mask_board, SudokuSolver
from sudoku_solver_csp_iterative import SudokuSolverCsp as SudokuSolverIterative
from enum import Enum, auto
import abc


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
    def __init__(self, parallel=False):
        self._parallel = parallel

    def solve_sudoku(self, board):
        sudoku_solver = SudokuSolverIterative(board)
        return sudoku_solver.solve(parallel=self._parallel)


class BenchmarkTestRunner:
    def __init__(self, board_size, num_runs=10):
        self._board_size = board_size
        self._num_runs = num_runs
        self._algorithm_runners = None
        self._algo_type_name = None
        self._board = get_solved_board(self._board_size)

    def set_algorithm_runners(self, algo_type):
        self._algo_type_name = algo_type.name.lower()

        if algo_type == Algorithm.CSP_RECURSIVE:
            self._algorithm_runners = RecursiveCspAlgorithmRunner()
        elif algo_type == Algorithm.CSP_ITERATIVE_SEQUENTIAL:
            self._algorithm_runners = IterativeCspAlgorithmRunner(parallel=False)
        elif algo_type == Algorithm.CSP_ITERATIVE_PARALLEL:
            self._algorithm_runners = IterativeCspAlgorithmRunner(parallel=True)
        elif algo_type == Algorithm.BRUTE_FORCE:
            self._algorithm_runners = BruteForceAlgorithmRunner()

    def run_benchmark(self, description=None, use_seed=False):
        if description is None:
            description = f"{self._board_size}x{self._board_size} benchmark"
        total_elapsed_time = 0
        failure_count = 0
        total_failure_time = 0

        for i in range(self._num_runs):
            start_time = time.time()
            masked_board = mask_board(original_board=self._board, seed=i) \
                if use_seed else mask_board(original_board=self._board)
            result = self._algorithm_runners.solve_sudoku(masked_board)
            end_time = time.time()
            elapsed_time = end_time - start_time

            print(f"test {i} result\n{result}\n")

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
    CSP_ITERATIVE_SEQUENTIAL = auto()
    CSP_ITERATIVE_PARALLEL = auto()


def main():
    board_size = 9
    benchmark_test_runner = BenchmarkTestRunner(board_size, num_runs=20)

    # benchmark_test_runner.set_algorithm_runners(Algorithm.BRUTE_FORCE)
    # benchmark_test_runner.run_benchmark()

    benchmark_test_runner.set_algorithm_runners(Algorithm.CSP_ITERATIVE_SEQUENTIAL)
    benchmark_test_runner.run_benchmark(use_seed=True)

    benchmark_test_runner.set_algorithm_runners(Algorithm.CSP_ITERATIVE_PARALLEL)
    benchmark_test_runner.run_benchmark(use_seed=True)

    # benchmark_test_runner.set_algorithm_runners(Algorithm.CSP_RECURSIVE)
    # benchmark_test_runner.run_benchmark()


if __name__ == '__main__':
    main()
