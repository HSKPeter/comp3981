from algo import *
import copy
from puzzle_loader import PuzzleLoader
from log_util import logger, log_format
from datetime import datetime
import uuid
import os
import time
from slack_alert import AlertSender
import argparse


# logger.remove()

now = datetime.now()
formatted_date_time = now.strftime('%Y-%m-%d_%H%M')

package_directory = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(package_directory, "log", f"{formatted_date_time}_exploration_{uuid.uuid4().hex}.log")
logger.add(log_file_path, format=log_format)


# def config_logger():
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
#
#     now = datetime.now()
#     formatted_date_time = now.strftime('%Y-%m-%d_%H%M')
#
#     package_directory = os.path.dirname(os.path.abspath(__file__))
#     log_file_path = os.path.join(package_directory, "log", f"{formatted_date_time}_exploration_{uuid.uuid4().hex}.log")
#
#     file_handler = logging.FileHandler(log_file_path, mode='a')
#     file_handler.setLevel(logging.INFO)
#     file_handler.flush = file_handler.stream.flush
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s\n %(message)s\n')
#     file_handler.setFormatter(formatter)
#
#     logger.addHandler(file_handler)
#     return logger


class SolvableSamplesExplorer:
    _package_directory = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, board):
        self._board = board
        self._size = len(self._board)
        self._algorithm_runner = None
        self._algo_type_name = None

    def explore(self, algorithm_type):
        if algorithm_type == AlgorithmType.BRUTE_FORCE:
            self._algorithm_runner = BruteForceAlgorithmRunner()
        elif algorithm_type == AlgorithmType.CSP_RECURSIVE:
            self._algorithm_runner = RecursiveCspAlgorithmRunner()
        elif algorithm_type == AlgorithmType.CSP_ITERATIVE:
            self._algorithm_runner = IterativeCspAlgorithmRunner()
        elif algorithm_type == AlgorithmType.CSP_ITERATIVE_MULTIPROCESS:
            self._algorithm_runner = IterativeCspAlgorithmRunner(is_parallel=True)
        else:
            raise ValueError("Invalid algorithm type")

        self._algo_type_name = algorithm_type.name.lower()
        board_copy = copy.deepcopy(self._board)
        return self._algorithm_runner.solve_sudoku(board_copy)

    def write_report(self, solvable_board, is_dev=True):
        path_directory = os.path.join(self._package_directory, "assets", "solvable_puzzle",
                                      f"{self._size}x{self._size}")

        samples_count = len(
            [name for name in os.listdir(path_directory) if os.path.isfile(os.path.join(path_directory, name))])
        index_of_new_sample = str(samples_count + 1).zfill(2)
        dev_flag = "_dev" if is_dev else ""
        file_path = os.path.join(self._package_directory, "assets", "solvable_puzzle", f"{self._size}x{self._size}",
                                 f"{self._size}x{self._size}_sample_{index_of_new_sample}{dev_flag}.txt")

        with open(file_path, 'a') as file:
            file.write(self.board_to_standard_text_format(solvable_board))

    @staticmethod
    def board_to_standard_text_format(board):
        num_strings_having_int_with_comma_separated = []
        for row in board:
            num_strings_having_int_with_comma_separated.append(",".join([str(i) for i in row]))

        return "\n".join(num_strings_having_int_with_comma_separated)


def clear_all_node_json_files():
    path_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nodes")
    file_names = [file_name for file_name in os.listdir(path_directory) if file_name != ".gitignore"]
    for file_name in file_names:
        file_path = os.path.join(path_directory, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

    return len(file_names)


def main():
    # board_size = 100

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", help="algorithm", type=str, default="i")
    parser.add_argument("-s", "--size", help="Board size", type=int, default=9)
    parser.add_argument("-i", "--index", help="Puzzle index", type=int, default=1)
    args = parser.parse_args()

    algo_type = None
    if args.algorithm == "b":
        algo_type = AlgorithmType.BRUTE_FORCE
    elif args.algorithm == "r":
        algo_type = AlgorithmType.CSP_RECURSIVE
    elif args.algorithm == "i":
        algo_type = AlgorithmType.CSP_ITERATIVE
    elif args.algorithm == "p":
        algo_type = AlgorithmType.CSP_ITERATIVE_MULTIPROCESS

    if args.size in [9, 12, 16, 25, 100]:
        board_puzzle_loader = PuzzleLoader()
        _, _, board_puzzle = board_puzzle_loader.load_unsolved_puzzle(size=args.size, sample_index=args.index)
    else:
        raise ValueError("Invalid board size")

    alert_sender = AlertSender()

    logger.info(f"Puzzle: {board_puzzle}")
    board_size = len(board_puzzle)
    start_msg = f"Starting exploration of puzzle (size: {board_size}x{board_size}) (algorithm: {algo_type})"
    logger.info(start_msg)
    alert_sender.send(start_msg)

    start_time = time.time()

    explorer = SolvableSamplesExplorer(board_puzzle)
    board_solved = explorer.explore(algo_type)

    end_time = time.time()
    elapsed_time = end_time - start_time

    logger.info(f"Solution found in {elapsed_time} seconds\n{board_solved}\n")
    alert_sender.send(f"Solution found in {elapsed_time} seconds")

    # if board_solved is not None:
    #     explorer.write_report(board_puzzle, is_dev=True)

    json_files_count = clear_all_node_json_files()
    logger.info(f"Deleted {json_files_count} node json files in nodes directory")


if __name__ == '__main__':
    main()
