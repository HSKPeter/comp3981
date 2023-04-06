import time
from datetime import datetime
from puzzle_loader import PuzzleLoader
from misc.algo_runner import *
import os
import uuid


class BenchmarkTestRunner:
    _package_directory = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, board_size=9, num_runs=10):
        self._board_size = board_size
        self._num_runs = num_runs
        self._algorithm_runners = None
        self._algo_type_name = None

    def set_algorithm_runners(self, algo_type):
        self._algo_type_name = algo_type.name.lower()

        if algo_type == AlgorithmType.CSP_RECURSIVE:
            self._algorithm_runners = RecursiveCspAlgorithmRunner()
        elif algo_type == AlgorithmType.CSP_ITERATIVE:
            self._algorithm_runners = IterativeCspAlgorithmRunner()
        elif algo_type == AlgorithmType.BRUTE_FORCE:
            self._algorithm_runners = BruteForceAlgorithmRunner()
        elif algo_type == AlgorithmType.CSP_ITERATIVE_MULTIPROCESS:
            self._algorithm_runners = IterativeCspAlgorithmRunner(is_parallel=True)

    def run_benchmark(self, board, algo_type):
        self.set_algorithm_runners(algo_type)
        return self.solve_board(board)

    def solve_board(self, board):
        start_time = time.time()
        result = self._algorithm_runners.solve_sudoku(board)
        end_time = time.time()
        elapsed_time = end_time - start_time

        message = f"Solution found by {self._algo_type_name} in {elapsed_time}: \n{result}\n"
        print(message)

        return message

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

        self.write_report(result, filename=self._algo_type_name)

    def write_report(self, content, filename="benchmarkreport", uuid_in_filename=False):
        now = datetime.now()
        filename_uuid = f"_{uuid.uuid4().hex}" if uuid_in_filename else ""
        formatted_date_time = now.strftime('%Y-%m-%d_%H%M')
        file_path = os.path.join(self._package_directory, "benchmark_reports", f"{formatted_date_time}_{filename}{filename_uuid}.txt")
        with open(file_path, 'a') as file:
            file.write(content)


def main():
    benchmark_test_runner = BenchmarkTestRunner()
    puzzle_loader = PuzzleLoader()

    report_content = ""

    sizes = [9, 12, 16, 25]
    for size in sizes:
        for sample_index in range(1, 4):

            file_path, raw_content, board = puzzle_loader.load_unsolved_puzzle(size,
                                                                               sample_index=sample_index,
                                                                               is_easy=True)

            report_content = f"Loaded puzzle from {file_path}:\n{raw_content}\n\n"

    print(report_content)

    algo_types_to_test = [
        AlgorithmType.BRUTE_FORCE,
        AlgorithmType.CSP_RECURSIVE,
        AlgorithmType.CSP_ITERATIVE,
        AlgorithmType.CSP_ITERATIVE_MULTIPROCESS,
    ]

    for type_to_test in algo_types_to_test:
        report_content += benchmark_test_runner.run_benchmark(board, type_to_test)


    benchmark_test_runner.write_report(report_content, uuid_in_filename=True)


if __name__ == '__main__':
    main()
