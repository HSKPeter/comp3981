import time
import numpy as np
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


def main():
    n = 50  # Number of times to run each algorithm
    performance_metrics = test_algorithms(NINE_X_NINE_SOLVED, n)
    for algo_name, metrics in performance_metrics.items():
        print(f"{algo_name}:")
        for metric_name, value in metrics.items():
            print(f" {metric_name}: {value:.5}")


if __name__ == '__main__':
    main()
