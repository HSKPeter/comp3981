import time
import numpy as np

import sudoku_solver
from sudoku_solver_csp import Assignments, Constraints, backtrack
from sudoku_solver import mask_board, SudokuSolver

NINE_X_NINE_SOLVED = [[7, 6, 2, 4, 9, 8, 5, 1, 3],
                      [9, 3, 1, 2, 5, 6, 4, 8, 7],
                      [4, 5, 8, 1, 3, 7, 2, 6, 9],
                      [5, 1, 9, 7, 8, 2, 3, 4, 6],
                      [6, 8, 7, 9, 4, 3, 1, 5, 2],
                      [3, 2, 4, 5, 6, 1, 9, 7, 8],
                      [2, 9, 6, 8, 1, 4, 7, 3, 5],
                      [8, 4, 5, 3, 7, 9, 6, 2, 1],
                      [1, 7, 3, 6, 2, 5, 8, 9, 4]]

TWENTY_FIVE_X_TWENTY_FIVE_SOLVED = [
    [23, 15, 16, 4, 18, 14, 2, 22, 5, 12, 20, 7, 8, 1, 10, 25, 11, 9, 17, 21, 6, 19, 24, 3, 13],
    [13, 19, 10, 1, 25, 23, 3, 7, 6, 17, 5, 2, 24, 18, 9, 20, 4, 14, 22, 15, 12, 16, 11, 21, 8],
    [8, 11, 21, 2, 6, 24, 15, 18, 1, 10, 12, 14, 17, 13, 19, 16, 3, 23, 5, 7, 22, 4, 25, 9, 20],
    [9, 22, 5, 12, 3, 4, 20, 21, 13, 8, 23, 16, 11, 15, 25, 2, 19, 6, 24, 18, 10, 7, 17, 14, 1],
    [17, 14, 24, 7, 20, 16, 11, 19, 9, 25, 22, 6, 4, 21, 3, 12, 13, 1, 10, 8, 23, 5, 15, 18, 2],
    [21, 2, 6, 8, 11, 10, 24, 1, 15, 18, 13, 12, 14, 19, 17, 3, 5, 7, 16, 23, 4, 22, 9, 20, 25],
    [5, 12, 3, 9, 22, 8, 4, 13, 20, 21, 15, 23, 16, 25, 11, 19, 24, 18, 2, 6, 7, 10, 14, 1, 17],
    [24, 7, 20, 17, 14, 25, 16, 9, 11, 19, 21, 22, 6, 3, 4, 13, 10, 8, 12, 1, 5, 23, 18, 2, 15],
    [16, 4, 18, 23, 15, 12, 14, 5, 2, 22, 1, 20, 7, 10, 8, 11, 17, 21, 25, 9, 19, 6, 3, 13, 24],
    [10, 1, 25, 13, 19, 17, 23, 6, 3, 7, 18, 5, 2, 9, 24, 4, 22, 15, 20, 14, 16, 12, 21, 8, 11],
    [3, 9, 12, 22, 5, 21, 13, 8, 4, 20, 11, 15, 25, 23, 16, 24, 18, 19, 6, 2, 17, 1, 7, 10, 14],
    [20, 17, 7, 14, 24, 19, 9, 25, 16, 11, 4, 21, 3, 22, 6, 10, 8, 13, 1, 12, 15, 2, 5, 23, 18],
    [25, 13, 1, 19, 10, 7, 6, 17, 23, 3, 24, 18, 9, 5, 2, 22, 15, 4, 14, 20, 11, 8, 16, 12, 21],
    [6, 8, 2, 11, 21, 18, 1, 10, 24, 15, 17, 13, 19, 12, 14, 5, 7, 3, 23, 16, 25, 20, 4, 22, 9],
    [18, 23, 4, 15, 16, 22, 5, 12, 14, 2, 8, 1, 10, 20, 7, 17, 21, 11, 9, 25, 24, 13, 19, 6, 3],
    [2, 6, 11, 21, 8, 1, 10, 15, 18, 24, 14, 19, 13, 17, 12, 23, 16, 5, 7, 3, 9, 25, 20, 4, 22],
    [12, 3, 22, 5, 9, 13, 8, 20, 21, 4, 16, 25, 15, 11, 23, 6, 2, 24, 18, 19, 14, 17, 1, 7, 10],
    [7, 20, 14, 24, 17, 9, 25, 11, 19, 16, 6, 3, 21, 4, 22, 1, 12, 10, 8, 13, 18, 15, 2, 5, 23],
    [4, 18, 15, 16, 23, 5, 12, 2, 22, 14, 7, 10, 1, 8, 20, 9, 25, 17, 21, 11, 3, 24, 13, 19, 6],
    [1, 25, 19, 10, 13, 6, 17, 3, 7, 23, 2, 9, 18, 24, 5, 14, 20, 22, 15, 4, 21, 11, 8, 16, 12],
    [22, 5, 9, 3, 12, 20, 21, 4, 8, 13, 25, 11, 23, 16, 15, 18, 6, 2, 19, 24, 1, 14, 10, 17, 7],
    [14, 24, 17, 20, 7, 11, 19, 16, 25, 9, 3, 4, 22, 6, 21, 8, 1, 12, 13, 10, 2, 18, 23, 15, 5],
    [19, 10, 13, 25, 1, 3, 7, 23, 17, 6, 9, 24, 5, 2, 18, 15, 14, 20, 4, 22, 8, 21, 12, 11, 16],
    [11, 21, 8, 6, 2, 15, 18, 24, 10, 1, 19, 17, 12, 14, 13, 7, 23, 16, 3, 5, 20, 9, 22, 25, 4],
    [15, 16, 23, 18, 4, 2, 22, 14, 12, 5, 10, 8, 20, 7, 1, 21, 9, 25, 11, 17, 13, 3, 6, 24, 19]]


def benchmark_algos():
    n = 50  # Number of times to run each algorithm
    performance_metrics = test_algorithms(NINE_X_NINE_SOLVED, n)
    for algo_name, metrics in performance_metrics.items():
        print(f"{algo_name}:")
        for metric_name, value in metrics.items():
            print(f" {metric_name}: {value:.5}")


def test_algorithms(solved_board, n):
    first_algo_times = []
    second_algo_times = []

    for _ in range(n):
        board = mask_board(solved_board)

        # First algorithm
        start_time = time.time()
        assignments = Assignments(board)
        constraints = Constraints(assignments)
        first_result = backtrack(constraints, assignments)
        first_algo_times.append(time.time() - start_time)

        # Second algorithm
        start_time = time.time()
        solver = SudokuSolver(board)
        solution = solver.solve()
        if solution is not None:
            second_result = solution.board
        second_algo_times.append(time.time() - start_time)

    first_algo_times = np.array(first_algo_times)
    second_algo_times = np.array(second_algo_times)

    performance_metrics = {
        "CSP Algorithm": {
            "min": np.min(first_algo_times),
            "max": np.max(first_algo_times),
            "mean": np.mean(first_algo_times),
            "variance": np.var(first_algo_times),
        },
        "Brute Force Algorithm": {
            "min": np.min(second_algo_times),
            "max": np.max(second_algo_times),
            "mean": np.mean(second_algo_times),
            "variance": np.var(second_algo_times),
        },
    }
    return performance_metrics


def time_function_and_log_to_file(func, file_name, num_runs, description):
    total_elapsed_time = 0
    failure_count = 0
    total_failure_time = 0

    for i in range(num_runs):
        start_time = time.time()
        result = func(i)
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"test {i} result\n{result}\n")

        if result is None:
            failure_count += 1
            total_failure_time += elapsed_time
        else:
            total_elapsed_time += elapsed_time

    successful_runs = num_runs - failure_count
    average_elapsed_time = total_elapsed_time / successful_runs if successful_runs > 0 else 0
    average_failure_time = total_failure_time / failure_count if failure_count > 0 else 0
    result = f"{func.__name__}\ndescription: {description}\n" \
             f"average runtime (seconds) over {successful_runs} successful runs: {average_elapsed_time:.6f}\n" \
             f"failures: {failure_count} (average time for failures: {average_failure_time:.6f})\n\n"
    print(result)
    with open(file_name, 'a') as file:
        file.write(result)


def run_csp(seed=None):
    board = mask_board(TWENTY_FIVE_X_TWENTY_FIVE_SOLVED, seed=seed)
    assignments = Assignments(board)
    constraints = Constraints(assignments)
    return backtrack(constraints, assignments, mute=True)


def main():
    time_function_and_log_to_file(run_csp, "csp_test_results.txt", 10, "25x25 benchmark")
    # benchmark_algos()


if __name__ == '__main__':
    main()
