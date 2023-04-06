import csv
import os
import time
from sudoku_solver_csp_iterative import solve_with_csp_iterative as solve_sudoku


def read_sudoku_from_file(file_path):
    with open(file_path, 'r') as file:
        sudoku = []
        for line in file:
            if line.strip() == '':
                continue
            row = list(map(int, line.strip().split(',')))
            sudoku.append(row)
    return sudoku


def solve_and_time(sudoku, solve_sudoku, n=1):
    total_time = 0
    for _ in range(n):
        start_time = time.time()
        solve_sudoku(sudoku)
        end_time = time.time()
        total_time += end_time - start_time
    return total_time / n


def process_sudoku_files(folder_path, solve_sudoku, n=1):
    results = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            sudoku = read_sudoku_from_file(file_path)
            avg_time = solve_and_time(sudoku, solve_sudoku, n)
            results.append((file_name, avg_time))
    return results


def process_folders(folder_paths, solve_sudoku, n=1):
    rows = []

    for folder_path in folder_paths:
        results = process_sudoku_files(folder_path, solve_sudoku, n)
        header = ['folder'] + [f for f, _ in results]
        row = [folder_path] + [avg_time for _, avg_time in results]
        rows.append(row)

    with open("timing_results.csv", "w", newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(header)
        csv_writer.writerows(rows)


def main():
    # folder_paths = [
    #     r"C:\Users\zohoo\Downloads\AllSamples\Sudoku-9x9",
    #     r"C:\Users\zohoo\Downloads\AllSamples\Sudoku-12x12",
    #     r"C:\Users\zohoo\Downloads\AllSamples\Sudoku-16x16",
    #     r"C:\Users\zohoo\Downloads\AllSamples\Sudoku-25x25",
    # ]
    # n = 1
    # process_folders(folder_paths, solve_sudoku, n)

    sample = r"C:\Users\zohoo\Downloads\AllSamples\Sudoku-25x25\25x25_sample1_grp1.txt"
    board = read_sudoku_from_file(sample)
    results = solve_sudoku(board)


if __name__ == "__main__":
    main()
