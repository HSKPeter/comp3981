from algo import *
import copy
from puzzle_loader import PuzzleLoader
from log_util import logger, log_format
from datetime import datetime
import uuid
import os
import time
from slack_alert import AlertSender
from board_sample_manager import BoardSampleManager

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
            file.write(BoardSampleManager.board_to_standard_text_format(solvable_board))


def main():
    # board_size = 100

    board_puzzle_loader = PuzzleLoader()
    _, _, board_puzzle = board_puzzle_loader.load_unsolved_puzzle(size=9, sample_index=1)
    alert_sender = AlertSender()

    # board_puzzle = [[36, 0, 0, 78, 0, 0, 0, 0, 0, 25, 0, 24, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 94, 0, 0, 28, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0, 0, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 7, 0, 0, 0, 0, 0, 0, 0, 0, 53, 0, 0, 0, 0, 0, 81, 93, 0], [0, 92, 0, 0, 0, 0, 91, 90, 0, 0, 0, 0, 0, 0, 0, 61, 0, 0, 0, 9, 0, 5, 0, 0, 0, 41, 0, 0, 0, 0, 95, 25, 57, 0, 0, 78, 0, 0, 0, 0, 88, 0, 0, 85, 0, 0, 32, 93, 37, 0, 0, 0, 0, 15, 13, 0, 0, 0, 0, 0, 0, 0, 1, 60, 0, 0, 0, 49, 0, 0, 75, 94, 0, 0, 0, 0, 0, 39, 0, 0, 0, 0, 0, 0, 14, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 51, 0, 93, 0, 18, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 0, 1, 0, 11, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 39, 0, 75, 0, 0, 0, 16, 0, 0, 91, 92, 0, 54, 0, 0, 0, 0, 96, 89, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 47, 0, 2, 0, 41, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 84, 0, 0, 50, 0, 73, 0, 71, 0, 0, 0, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 13, 0, 92, 68, 0, 91, 0, 35, 0, 0, 0, 34, 0, 0, 0, 0, 22, 0, 0, 96, 0], [0, 0, 71, 0, 0, 0, 69, 0, 0, 0, 0, 0, 0, 78, 36, 27, 0, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 21, 0, 44, 0, 0, 0, 1, 0, 10, 60, 0, 0, 0, 0, 0, 0, 97, 0, 0, 0, 0, 0, 90, 68, 0, 0, 54, 91, 67, 0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 83, 0, 0, 0, 0, 0, 0, 0, 0, 0, 53, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 64], [0, 89, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 56, 0, 91, 68, 0, 0, 0, 78, 0, 0, 55, 0, 0, 0, 0, 0, 0, 0, 71, 0, 0, 0, 0, 0, 0, 7, 0, 0, 17, 0, 0, 0, 0, 0, 93, 0, 0, 0, 0, 18, 0, 0, 85, 0, 0, 62, 0, 23, 0, 0, 21, 0, 3, 0, 0, 0, 74, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 52, 6, 0, 0, 0, 0, 0, 0, 0, 0, 31, 0, 0, 48], [0, 0, 0, 0, 0, 72, 0, 0, 42, 63, 0, 0, 31, 0, 83, 48, 66, 0, 0, 30, 0, 17, 0, 0, 0, 0, 77, 0, 0, 0, 0, 81, 0, 0, 85, 0, 0, 0, 88, 0, 0, 23, 33, 0, 0, 94, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 1, 74, 0, 0, 0, 9, 0, 0, 0, 0, 34, 96, 0, 0, 0, 0, 0, 56, 70, 67, 68, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0], [0, 0, 0, 0, 0, 0, 0, 93, 0, 0, 8, 0, 15, 0, 0, 0, 0, 29, 0, 0, 0, 0, 0, 0, 96, 0, 46, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 35, 70, 0, 0, 0, 0, 0, 0, 42, 0, 0, 26, 0, 0, 0, 0, 0, 14, 0, 0, 0, 83, 0, 0, 0, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 57, 0, 0, 0, 43, 0, 0, 0, 0, 76, 0, 0, 0, 0, 0, 1, 0, 0, 62, 0, 94, 0, 0, 87, 0, 0], [80, 15, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 0, 88, 85, 0, 32, 0, 53, 0, 56, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 61, 0, 0, 0, 0, 22, 0, 0, 0, 48, 66, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 99, 0, 0, 0, 27, 43, 0, 0, 0, 0, 0, 0, 0, 69, 0, 0, 0, 0, 0, 0, 73, 0, 0, 21, 23, 3, 0, 0, 0, 28, 0, 75, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0], [0, 21, 75, 0, 0, 0, 0, 0, 0, 87, 0, 0, 0, 0, 74, 0, 0, 44, 11, 0, 0, 0, 50, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 26, 6, 0, 0, 0, 0, 43, 47, 0, 25, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 77, 0, 0, 13, 8, 0, 0, 0, 0, 37, 81, 0, 32, 0, 0, 0, 0, 9, 0, 46, 97, 0, 0, 0, 0, 0, 68, 0, 0, 0, 0, 92, 56, 90, 16], [87, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 53, 0, 88, 0, 0, 0, 51, 0, 0, 0, 68, 0, 66, 0, 35, 91, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 83, 0, 0, 48, 0, 14, 30, 0, 0, 100, 0, 0, 0, 0, 0, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 86, 81, 0, 0, 0, 37, 0, 0, 0, 0, 94, 0, 0, 0, 0, 0, 0, 0, 0, 60, 49, 56, 0, 10, 0, 0, 76, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 0, 0, 0, 0, 27, 0, 0, 0, 0, 0, 40, 0, 0, 47, 0, 0, 0, 0, 0, 0, 32, 24, 81, 0, 87, 0, 0, 80, 0, 8, 0, 39, 0, 7, 44, 51, 4, 0, 0, 0, 85, 0, 88, 58, 0, 0, 62, 33, 0, 28, 94, 0, 0, 0, 74, 0, 0, 0, 0, 0, 0, 0, 49, 0, 99, 0, 0, 72, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 95, 0], [0, 0, 83, 30, 0, 0, 0, 0, 48, 0, 0, 0, 0, 0, 5, 52, 26, 0, 0, 19, 0, 0, 0, 0, 0, 58, 93, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 76, 10, 0, 67, 0, 0, 0, 33, 0, 0, 0, 28, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 91, 31, 0, 0, 0, 96, 0, 42, 46, 0, 0, 0, 0, 0, 0, 0, 47, 0, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 41], [0, 0, 0, 0, 37, 24, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 27, 40, 0, 75, 3, 0, 23, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 74, 0, 67, 0, 56, 0, 0, 0, 0, 0, 0, 96, 61, 0, 0, 0, 45, 0, 31, 0, 68, 0, 0, 0, 35, 0, 99, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 0, 0, 0, 0, 0, 93, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 72, 100, 0, 0, 0, 0, 0, 0, 57, 0, 0, 0, 0, 0, 0, 84, 0, 0, 0, 0, 7, 0, 0, 8, 0, 77, 0, 0, 0, 0, 0, 0, 53, 0, 51, 58, 0, 0, 33, 0, 0, 3, 0, 0, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0, 74, 0, 0, 0, 65, 22, 34, 0, 46, 0, 9, 0, 0, 0, 0, 45, 0, 54, 31, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 43, 0, 0, 27, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 10, 67, 0, 0, 12, 0, 0, 23, 0, 62, 3, 75, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 59, 0, 0, 0, 30, 0, 0, 25, 0, 0, 0, 0, 0, 0, 0, 71, 0, 0, 0, 0, 47, 0, 0, 0, 0, 0, 0, 0, 17, 85, 0, 0, 51, 0, 0, 18, 0, 0, 44, 80, 0, 87, 39, 0, 13, 0, 0, 79, 7, 0, 0, 0, 68, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 0, 0, 65], [0, 18, 0, 53, 0, 51, 0, 0, 0, 82, 39, 0, 77, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 42, 0, 0, 0, 46, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 31, 0, 0, 0, 0, 0, 0, 0, 20, 98, 0, 0, 0, 0, 0, 0, 0, 48, 0, 0, 25, 0, 81, 0, 0, 0, 86, 0, 0, 73, 0, 0, 40, 0, 0, 0, 0, 15, 0, 0, 43, 1, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 94, 0, 0, 62], [0, 0, 0, 0, 0, 33, 0, 0, 75, 0, 67, 0, 0, 0, 0, 74, 0, 1, 10, 0, 48, 59, 0, 0, 0, 0, 0, 0, 0, 0, 99, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 36, 0, 40, 0, 29, 0, 0, 0, 0, 0, 0, 0, 86, 0, 0, 0, 0, 0, 0, 80, 0, 0, 0, 77, 0, 0, 0, 0, 0, 58, 44, 0, 0, 0, 0, 0, 53, 61, 0, 0, 0, 0, 0, 63, 0, 0, 0, 68, 0, 0, 0, 90, 0, 0, 0, 0, 35], [0, 55, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 76, 0, 0, 0, 56, 0, 0, 11, 0, 0, 0, 0, 0, 62, 23, 12, 0, 97, 0, 0, 0, 31, 0, 0, 90, 16, 0, 0, 0, 42, 0, 38, 0, 0, 0, 61, 34, 0, 63, 48, 0, 0, 0, 0, 0, 0, 0, 30, 0, 52, 6, 0, 0, 0, 0, 0, 0, 19, 0, 0, 77, 0, 7, 8, 0, 0, 39, 21, 0, 53, 0, 0, 0, 93, 0, 0, 0, 0, 0], [0, 91, 0, 0, 0, 70, 0, 0, 0, 54, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 73, 32, 81, 0, 41, 0, 0, 0, 0, 55, 40, 0, 43, 0, 0, 17, 0, 58, 0, 0, 88, 82, 0, 0, 0, 18, 53, 0, 0, 21, 77, 0, 8, 0, 0, 0, 87, 0, 0, 0, 0, 76, 0, 11, 0, 60, 0, 0, 0, 0, 12, 0, 0, 0, 0, 33, 0, 0, 14, 50, 30, 84, 0, 0, 0, 0, 0, 72, 19, 0, 99, 6, 0, 0, 0, 0, 0], [16, 49, 0, 56, 0, 0, 0, 0, 10, 74, 0, 9, 33, 0, 0, 0, 0, 28, 0, 34, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 83, 0, 0, 95, 25, 0, 0, 0, 0, 0, 0, 53, 32, 0, 0, 0, 0, 0, 81, 0, 0, 0, 47, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 59, 0, 31, 0, 0, 0, 30, 0, 0, 0, 0, 0, 38, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 60, 0, 58, 1, 0, 0, 0, 0, 0, 0, 0, 0, 66, 0, 48, 31, 45, 0, 59, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 83, 0, 84, 0, 0, 0, 69, 2, 0, 0, 5, 0, 0, 0, 0, 71, 40, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 86, 82, 81, 28, 0, 34, 0, 0, 62, 61, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 98, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 7, 0, 0, 0, 15, 0, 0, 0, 86, 0, 0, 0, 0, 0, 0, 0, 0, 53, 75, 0, 0, 0, 0, 21, 0, 94, 0, 87, 0, 0, 0, 0, 58, 0, 93, 0, 0, 0, 28, 61, 0, 0, 62, 0, 0, 0, 0, 46, 67, 92, 0, 0, 49, 0, 0, 10, 0, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 83, 0, 0], [0, 24, 0, 0, 53, 0, 37, 0, 0, 41, 0, 0, 0, 0, 64, 29, 0, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 12, 0, 34, 0, 0, 49, 92, 0, 0, 91, 54, 16, 0, 0, 0, 0, 42, 0, 0, 0, 0, 22, 63, 0, 59, 0, 0, 0, 0, 0, 0, 45, 0, 0, 0, 0, 2, 52, 0, 0, 73, 5, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 75, 8, 21, 0, 0, 0, 94, 39], [0, 0, 45, 0, 0, 0, 31, 0, 0, 0, 0, 0, 0, 0, 99, 42, 0, 96, 0, 26, 0, 0, 53, 0, 0, 0, 0, 32, 24, 0, 0, 0, 0, 0, 29, 0, 0, 0, 0, 0, 0, 76, 60, 0, 0, 0, 93, 0, 0, 0, 94, 3, 23, 0, 0, 0, 0, 80, 39, 75, 0, 0, 0, 54, 0, 0, 0, 0, 0, 91, 0, 0, 0, 0, 0, 0, 0, 0, 34, 97, 0, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20], [0, 0, 44, 0, 60, 76, 0, 0, 0, 0, 0, 0, 0, 87, 0, 0, 0, 0, 0, 0, 0, 65, 0, 0, 98, 0, 0, 0, 0, 26, 0, 35, 0, 0, 0, 66, 0, 0, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 78, 43, 0, 0, 0, 84, 83, 0, 0, 86, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 92, 0, 0, 0, 68, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0], [0, 0, 20, 0, 0, 0, 0, 0, 0, 52, 55, 0, 0, 0, 0, 95, 83, 84, 0, 0, 0, 0, 0, 0, 94, 75, 21, 0, 0, 0, 0, 88, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 46, 33, 97, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 26, 0, 38, 22, 72, 63, 98, 0, 0, 0, 14, 0, 0, 0, 0, 59, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0], [61, 0, 0, 97, 0, 0, 89, 0, 0, 0, 0, 68, 0, 56, 16, 0, 0, 0, 92, 54, 0, 0, 43, 25, 0, 0, 0, 0, 50, 78, 0, 52, 19, 0, 0, 0, 69, 0, 71, 73, 0, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 82, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 44, 0, 0, 0, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 0, 0, 0, 0, 0, 66, 0, 48, 0, 0, 30, 70, 35, 0, 0], [64, 0, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 81, 0, 0, 0, 0, 0, 0, 10, 0, 68, 0, 0, 0, 0, 0, 49, 0, 0, 0, 0, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 66, 0, 0, 0, 22, 0, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 0, 43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 23, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 57, 55, 0, 0, 0, 0, 19, 0, 0, 0, 52, 0, 0, 2, 0, 0, 0, 0, 0, 0, 4, 44, 51, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 23, 0, 54, 0, 0, 0, 92, 0, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 0, 45, 0, 0, 31, 0, 30, 66, 14, 0, 0, 0, 0, 0, 0, 0, 0, 26, 0, 40, 0, 0, 0, 0, 0, 0, 77, 0, 0, 0, 82, 0, 86, 0, 0, 0, 0, 0, 0], [19, 0, 0, 0, 0, 6, 0, 0, 0, 0, 25, 95, 0, 14, 50, 0, 0, 0, 0, 0, 0, 15, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 0, 0, 0, 58, 0, 0, 44, 0, 0, 0, 0, 0, 0, 3, 97, 0, 94, 0, 0, 67, 1, 11, 0, 76, 4, 60, 0, 34, 22, 0, 96, 0, 65, 0, 0, 0, 0, 0, 0, 70, 0, 16, 92, 45, 0, 90, 0, 0, 0, 86, 0, 0, 0, 0, 0, 32, 73, 55, 40, 0, 0, 0, 0, 0, 57, 17, 0], [51, 0, 0, 0, 44, 0, 0, 58, 82, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 34, 0, 0, 46, 0, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 66, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 84, 95, 0, 0, 0, 59, 0, 0, 50, 0, 24, 73, 0, 0, 0, 0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 67, 0, 94, 0, 0, 0, 0, 0, 0, 0, 97, 0], [0, 27, 0, 55, 29, 0, 0, 17, 78, 0, 81, 32, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 56, 0, 0, 0, 1, 0, 3, 0, 75, 62, 0, 0, 97, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 66, 0, 0, 0, 96, 0, 0, 0, 0, 0, 89, 0, 0, 59, 0, 0, 84, 31, 83, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 82, 0, 0, 0, 0, 58, 53], [0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 42, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 32, 69, 0, 0, 0, 0, 0, 0, 78, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 93, 0, 53, 0, 0, 82, 0, 0, 18, 0, 0, 0, 0, 0, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 56, 0, 0, 0, 0, 0, 21, 0, 0, 0, 94, 0, 0, 0, 14, 0, 0, 0, 0, 0, 30, 0, 0, 19, 0, 52, 0, 0, 38, 0, 0], [24, 71, 0, 0, 0, 86, 0, 0, 2, 0, 0, 0, 0, 0, 47, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 28, 76, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 89, 65, 34, 0, 0, 0, 66, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 48, 0, 0, 0, 0, 0, 0, 0, 93, 0, 0, 0, 0, 0, 0, 53, 0, 8, 79, 0, 80, 0, 0, 0, 0, 7], [0, 0, 7, 0, 0, 0, 80, 87, 0, 0, 0, 44, 0, 18, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0, 0, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 96, 0, 0, 0, 0, 0, 30, 0, 83, 59, 0, 0, 0, 0, 6, 0, 0, 0, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 62, 0, 33, 0, 0, 0, 0, 10, 49, 76, 0, 0, 0, 0, 0, 60], [0, 0, 9, 0, 0, 96, 0, 0, 0, 0, 0, 45, 0, 91, 0, 0, 0, 0, 0, 0, 78, 57, 0, 0, 0, 0, 0, 0, 0, 40, 0, 0, 0, 41, 0, 0, 81, 0, 24, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 58, 93, 0, 0, 0, 0, 0, 37, 0, 51, 3, 0, 23, 28, 0, 0, 0, 0, 94, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 99, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 83, 0, 0, 31, 0, 0], [0, 0, 0, 11, 67, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 62, 0, 0, 0, 20, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 83, 0, 0, 0, 0, 50, 95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 0, 0, 0, 43, 47, 0, 0, 0, 0, 0, 88, 0, 0, 0, 58, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 54, 0, 0, 0, 0, 92, 0, 66, 0, 68, 46, 96, 0, 0, 65, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 37, 0, 18, 0, 0, 0, 0, 0, 93, 13, 0, 0, 80, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 74, 0, 56, 1, 0, 0, 0, 0, 75, 0, 0, 3, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 45, 0, 66, 0, 0, 0, 0, 0, 0, 0, 0, 96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 69, 0, 0, 0, 0, 32, 0, 0, 81, 0], [0, 0, 0, 0, 12, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 0, 0, 0, 10, 0, 31, 0, 14, 25, 0, 83, 0, 0, 84, 0, 0, 99, 52, 0, 0, 0, 0, 0, 0, 0, 0, 29, 43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 71, 0, 0, 0, 0, 73, 24, 0, 79, 0, 0, 0, 0, 0, 39, 77, 0, 53, 88, 0, 0, 0, 37, 0, 82, 93, 0, 0, 0, 96, 0, 0, 0, 22, 0, 0, 9, 0, 0, 0, 54, 0, 0, 0, 0, 66, 0], [0, 0, 0, 0, 0, 0, 0, 53, 0, 0, 0, 0, 0, 0, 0, 0, 40, 0, 13, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 49, 0, 0, 0, 0, 67, 0, 16, 91, 0, 0, 99, 0, 0, 0, 26, 0, 0, 0, 0, 30, 0, 83, 0, 0, 0, 70, 90, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 0, 0, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 44, 0, 0, 11, 0, 0, 0, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 63, 0, 0, 0, 99, 0, 0, 0, 0, 30, 83, 66, 0, 0, 0, 0, 70, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 24, 0, 0, 0, 0, 0, 53, 0, 0, 0, 0, 0, 62, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 92, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 0, 78, 36, 0, 0, 0, 0], [0, 0, 0, 29, 0, 64, 0, 7, 47, 0, 0, 0, 0, 0, 18, 0, 0, 0, 82, 0, 0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 33, 0, 97, 0, 89, 12, 9, 0, 0, 65, 0, 48, 0, 0, 90, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 0, 69, 0, 5, 6, 41, 19, 0, 20, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 0, 0], [0, 56, 0, 0, 0, 16, 0, 68, 0, 10, 0, 65, 97, 12, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 0, 69, 2, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 0, 0, 85, 0, 0, 0, 0, 24, 0, 0, 0, 7, 0, 0, 17, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 0, 94, 23, 0, 8, 0, 0, 0, 39, 0, 0, 0, 0, 0, 90, 0, 0, 0, 31, 42, 0, 98, 0, 0, 0, 63, 0, 72, 0], [0, 5, 0, 20, 0, 0, 0, 0, 0, 0, 43, 0, 0, 0, 55, 0, 84, 0, 0, 0, 0, 0, 62, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58, 0, 0, 0, 0, 0, 11, 74, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 92, 0, 0, 0, 38, 99, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 0, 0, 24, 81, 0, 0, 0, 86, 0, 0, 88, 0, 0, 64, 0, 0, 13, 80, 17, 0, 7, 0], [0, 0, 0, 0, 65, 61, 0, 0, 0, 0, 0, 0, 56, 0, 0, 0, 10, 49, 0, 16, 0, 0, 0, 0, 0, 0, 78, 0, 25, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 53, 85, 0, 0, 0, 82, 0, 0, 0, 0, 79, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 0, 98, 0, 0, 0, 45, 48, 14, 70, 0, 0, 0, 0, 30, 0], [0, 0, 0, 0, 36, 27, 0, 0, 50, 0, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 79, 0, 0, 0, 0, 0, 23, 0, 94, 62, 0, 16, 0, 0, 10, 0, 0, 68, 0, 0, 0, 0, 0, 97, 0, 34, 0, 28, 0, 0, 0, 14, 31, 0, 90, 0, 0, 0, 0, 0, 0, 0, 0, 72, 63, 0, 0, 22, 0, 0, 47, 17, 0, 0, 0, 40, 0, 0, 80, 0, 0, 85, 0, 24, 0, 88, 0, 0, 0, 0], [0, 0, 0, 44, 0, 0, 0, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 75, 0, 0, 0, 0, 0, 98, 0, 0, 0, 0, 0, 0, 0, 0, 31, 45, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 0, 0, 82, 81, 0, 0, 0, 15, 0, 0, 7, 0, 40, 0, 47, 0, 0, 0, 56, 0, 67, 0, 0, 91, 0, 0, 0, 0, 0, 0, 33, 34, 0, 0, 0, 0, 0], [0, 0, 0, 45, 83, 48, 0, 30, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 0, 99, 0, 0, 88, 32, 0, 0, 0, 0, 0, 85, 0, 40, 0, 0, 15, 0, 0, 0, 0, 80, 0, 0, 74, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0, 94, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 95, 0, 0, 0, 0, 0, 57, 0, 71, 0, 0, 0, 0, 0, 6, 0, 0], [94, 87, 0, 39, 0, 75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 76, 0, 70, 0, 0, 0, 0, 0, 59, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 0, 0, 50, 0, 25, 0, 0, 0, 0, 0, 0, 0, 19, 6, 0, 0, 47, 0, 0, 0, 0, 0, 17, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 33, 97, 0, 0, 34, 0, 0, 0, 0, 0, 0, 16, 91, 0, 0, 0, 56, 0, 0, 92], [0, 0, 97, 0, 46, 89, 0, 61, 0, 0, 0, 91, 0, 0, 54, 56, 0, 0, 0, 0, 0, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 6, 20, 0, 0, 71, 100, 0, 0, 0, 0, 77, 17, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 7, 0, 0, 94, 0, 0, 58, 0, 76, 0, 0, 0, 0, 0, 4, 51, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 59, 0, 0, 0, 90, 0, 0, 66], [0, 93, 0, 51, 11, 4, 0, 0, 88, 0, 0, 94, 0, 79, 0, 0, 7, 0, 0, 21, 0, 0, 98, 0, 0, 0, 0, 63, 0, 0, 0, 68, 90, 0, 0, 0, 48, 0, 0, 0, 0, 0, 69, 0, 72, 0, 0, 0, 0, 0, 0, 57, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 29, 0, 0, 40, 0, 77, 0, 0, 47, 0, 0, 92, 0, 0, 0, 0, 0, 91, 56, 0, 0, 0, 0, 0, 46, 0, 0, 0, 97], [0, 86, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 92, 0, 0, 26, 0, 0, 0, 0, 42, 0, 99, 0, 0, 0, 0, 0, 0, 0, 45, 35, 0, 0, 0, 52, 0, 0, 0, 0, 0, 0, 0, 0, 71, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 84, 0, 0, 0, 30, 0, 95, 0, 0, 0, 0, 79, 0, 0, 0, 87, 0, 0, 0, 0, 93, 44, 58, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 62, 0, 0, 33, 16, 0, 0, 0, 49, 0, 0, 60, 0, 0, 65, 26, 0, 0, 0, 0, 96, 0, 22, 99, 0, 0, 0, 0, 90, 0, 14, 0, 0, 0, 41, 0, 0, 0, 32, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 77, 40, 0, 0, 0], [3, 0, 0, 0, 0, 21, 0, 0, 0, 7, 0, 11, 0, 0, 0, 0, 0, 0, 44, 0, 35, 68, 0, 70, 0, 59, 45, 0, 0, 31, 65, 9, 96, 0, 0, 22, 99, 0, 26, 0, 0, 0, 0, 0, 0, 95, 0, 27, 0, 50, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 17, 0, 43, 0, 0, 77, 0, 0, 81, 0, 82, 0, 0, 0, 0, 0, 37, 0, 0, 28, 89, 0, 12, 0, 0, 0, 0, 97, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0], [0, 96, 0, 0, 98, 38, 42, 0, 0, 0, 48, 0, 0, 0, 59, 0, 68, 35, 0, 0, 0, 0, 0, 0, 0, 13, 29, 0, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 82, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 51, 0, 0, 0, 0, 76, 62, 0, 0, 0, 0, 0, 0, 46, 0, 61, 56, 0, 0, 16, 0, 60, 0, 0, 0, 0, 0, 6, 100, 0, 0, 0, 2, 0, 69, 5, 50, 57, 0, 83, 0, 0, 84, 30, 0, 0], [0, 90, 66, 0, 0, 0, 0, 0, 0, 0, 99, 0, 0, 22, 0, 63, 0, 0, 0, 0, 41, 0, 18, 0, 0, 82, 0, 0, 0, 0, 0, 0, 40, 0, 0, 47, 0, 0, 0, 0, 0, 4, 0, 58, 53, 44, 0, 1, 0, 0, 75, 21, 94, 0, 0, 0, 80, 7, 0, 0, 74, 54, 0, 0, 0, 0, 10, 0, 0, 0, 97, 12, 0, 0, 28, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 0, 0, 0, 0, 0, 0, 69, 0, 0, 71, 5], [78, 84, 0, 0, 0, 57, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 56, 0, 67, 0, 0, 10, 49, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 59, 0, 0, 0, 0, 0, 14, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 13, 0, 0, 17, 24, 0, 0, 0, 32, 18, 0, 73, 0, 0], [54, 10, 0, 49, 0, 0, 67, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 62, 12, 0, 0, 72, 69, 0, 0, 0, 0, 5, 6, 100, 0, 0, 84, 95, 0, 50, 27, 0, 0, 55, 82, 0, 0, 0, 73, 0, 41, 0, 86, 0, 64, 0, 77, 0, 0, 0, 36, 43, 0, 0, 88, 0, 58, 0, 0, 44, 0, 0, 51, 0, 0, 39, 3, 0, 0, 0, 94, 0, 0, 0, 0, 0, 0, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 96, 0, 0, 0], [0, 0, 0, 0, 77, 0, 0, 64, 0, 0, 85, 18, 0, 0, 0, 0, 0, 41, 0, 0, 0, 60, 0, 0, 0, 0, 67, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 34, 0, 59, 0, 0, 0, 0, 45, 35, 0, 0, 0, 0, 0, 0, 96, 22, 42, 65, 0, 63, 0, 0, 0, 0, 57, 0, 95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 69, 0, 0, 19, 0, 0, 21, 79, 0, 0, 3, 0, 0, 0, 51, 4, 76, 0, 0, 0, 93, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 82, 85, 41, 86, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 74, 0, 0, 94, 62, 0, 33, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 66, 0, 59, 0, 0, 0, 0, 0, 0, 96, 0, 0, 46, 0, 0, 0, 0, 0, 95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 52, 98, 0, 0, 0, 0, 7, 0, 0, 0, 87, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51], [0, 0, 19, 0, 0, 20, 5, 0, 72, 0, 78, 0, 83, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 80, 0, 53, 0, 0, 58, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 28, 0, 0, 0, 0, 0, 56, 0, 11, 0, 0, 9, 38, 0, 0, 46, 0, 0, 0, 0, 0, 0, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 40, 0, 0, 43, 0, 0, 36, 0, 13, 0], [0, 0, 0, 10, 16, 0, 0, 0, 60, 0, 0, 0, 0, 28, 0, 33, 0, 0, 97, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 84, 78, 0, 0, 0, 37, 0, 85, 24, 69, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0, 17, 0, 0, 0, 0, 53, 0, 51, 44, 18, 0, 0, 0, 0, 0, 79, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 66, 0, 0, 0, 0, 0, 96, 0, 38, 0, 0, 99, 0, 46, 0, 0], [38, 0, 22, 0, 0, 0, 63, 0, 0, 0, 0, 0, 0, 0, 0, 70, 0, 0, 0, 45, 0, 0, 0, 40, 0, 0, 17, 0, 36, 29, 73, 0, 0, 0, 0, 86, 0, 32, 0, 0, 0, 0, 0, 79, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 4, 23, 0, 0, 12, 0, 0, 0, 0, 0, 34, 0, 0, 0, 0, 0, 0, 0, 0, 67, 0, 72, 0, 0, 0, 5, 98, 0, 0, 0, 19, 84, 95, 0, 0, 25, 0, 0, 0, 0, 50], [37, 41, 0, 0, 0, 32, 0, 0, 73, 0, 0, 0, 0, 0, 15, 0, 55, 0, 17, 0, 23, 0, 0, 0, 34, 0, 97, 0, 0, 0, 0, 11, 0, 0, 0, 10, 0, 0, 92, 0, 0, 0, 99, 22, 46, 63, 0, 0, 0, 0, 0, 45, 0, 0, 0, 0, 0, 91, 0, 0, 0, 0, 0, 0, 0, 0, 52, 0, 6, 0, 0, 0, 0, 78, 0, 0, 0, 0, 95, 0, 0, 88, 0, 93, 0, 18, 0, 0, 0, 0, 8, 39, 0, 0, 0, 0, 0, 0, 0, 79], [0, 83, 0, 0, 0, 95, 0, 0, 30, 14, 2, 0, 0, 6, 0, 0, 0, 72, 0, 0, 0, 18, 0, 0, 76, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 56, 0, 54, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 0, 89, 68, 0, 70, 45, 91, 66, 0, 0, 90, 0, 22, 0, 38, 0, 65, 0, 99, 9, 0, 96, 0, 0, 0, 0, 0, 55, 15, 0, 0, 0, 0, 0, 37, 73, 0, 85, 41, 69, 0, 0], [0, 0, 33, 0, 0, 12, 0, 34, 0, 0, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27, 0, 0, 57, 0, 0, 83, 95, 0, 0, 0, 0, 19, 0, 0, 20, 100, 0, 0, 0, 64, 47, 0, 0, 43, 0, 0, 40, 0, 0, 0, 41, 0, 0, 0, 0, 24, 37, 0, 21, 79, 0, 0, 87, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 65, 42, 0, 0, 0, 0, 0, 99, 0, 0, 45, 0, 0, 0, 0, 0, 0, 59, 0], [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 91, 0, 0, 70, 90, 0, 0, 0, 48, 100, 0, 71, 19, 98, 0, 0, 0, 0, 6, 0, 0, 27, 0, 84, 0, 0, 0, 50, 0, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 47, 0, 0, 0, 0, 0, 0, 43, 0, 40, 0, 0, 67, 0, 56, 0, 0, 54, 0, 0, 0, 0, 89, 0, 97, 0, 0, 0, 0, 33], [0, 0, 0, 0, 0, 39, 0, 3, 0, 0, 76, 0, 0, 0, 0, 0, 0, 0, 58, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 45, 0, 0, 0, 0, 22, 0, 0, 0, 38, 0, 57, 0, 27, 50, 0, 25, 0, 0, 83, 0, 0, 0, 0, 0, 0, 0, 72, 98, 0, 100, 0, 0, 0, 0, 55, 17, 0, 0, 0, 13, 0, 81, 37, 0, 0, 0, 0, 73, 0, 0, 0, 0, 0, 28, 0, 0, 0, 0, 61, 0, 10, 0, 92, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 66, 0, 0, 91, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 86, 0, 37, 81, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 44, 0, 0, 0, 0, 0, 0, 88, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 34, 0, 0, 0, 0, 0, 28, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 72, 5, 0, 0, 0, 0, 0], [0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 0, 0, 0, 0, 0, 0, 0, 0, 0, 43, 0, 0, 0, 0, 20, 2, 0, 0, 71, 0, 0, 0, 0, 0, 80, 0, 77, 0, 0, 0, 0, 0, 64, 0, 88, 0, 0, 0, 53, 0, 0, 18, 93, 21, 0, 94, 62, 0, 0, 0, 97, 0, 0, 0, 60, 0, 49, 0, 0, 0, 4, 74, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [84, 0, 14, 0, 0, 0, 0, 0, 0, 45, 0, 0, 26, 99, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 53, 0, 0, 0, 15, 0, 0, 0, 0, 0, 79, 0, 0, 0, 0, 0, 56, 0, 44, 0, 0, 49, 0, 0, 0, 0, 97, 0, 75, 0, 0, 0, 0, 0, 92, 0, 91, 35, 0, 0, 54, 0, 0, 0, 0, 0, 96, 0, 0, 12, 0, 89, 0, 61, 0, 78, 36, 0, 0, 0, 40, 0, 17, 0, 0, 0, 0, 100, 0, 0, 2, 0, 0, 0], [93, 82, 0, 85, 58, 0, 0, 0, 37, 0, 0, 0, 13, 0, 0, 0, 29, 0, 0, 0, 0, 12, 0, 61, 22, 0, 9, 0, 0, 0, 0, 0, 0, 68, 0, 16, 0, 0, 0, 0, 0, 52, 0, 0, 0, 72, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 84, 0, 86, 0, 0, 0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 76, 74, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 97, 0, 0, 33, 0], [10, 76, 0, 0, 56, 0, 60, 49, 4, 0, 0, 0, 3, 0, 0, 94, 0, 0, 0, 0, 0, 42, 0, 99, 0, 0, 0, 98, 0, 52, 31, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 69, 0, 73, 0, 0, 0, 0, 47, 0, 0, 0, 0, 43, 0, 0, 55, 40, 0, 93, 18, 0, 0, 53, 82, 0, 0, 0, 0, 0, 8, 79, 0, 0, 0, 0, 0, 64, 0, 0, 35, 0, 0, 0, 0, 70, 66, 0, 61, 65, 96, 0, 0, 63, 0, 12, 22, 0], [0, 0, 0, 0, 5, 52, 0, 0, 0, 0, 0, 0, 59, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 7, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 88, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 44, 0, 10, 89, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 0, 43, 17, 0, 95, 47, 55], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 78, 0, 0, 0, 95, 0, 43, 0, 0, 0, 0, 0, 0, 0, 23, 94, 3, 0, 4, 0, 0, 60, 0, 1, 0, 74, 0, 56, 0, 0, 0, 46, 0, 0, 0, 0, 0, 61, 0, 0, 0, 54, 0, 0, 0, 0, 0, 0, 0, 0, 98, 0, 0, 0, 26, 0, 0, 19, 0, 0, 0, 50, 0, 45, 25, 31, 0, 48, 0, 0, 88, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 79, 77], [40, 0, 55, 0, 0, 0, 0, 47, 0, 95, 24, 81, 0, 0, 0, 0, 20, 0, 73, 41, 0, 0, 0, 0, 49, 10, 60, 11, 0, 0, 0, 0, 3, 23, 94, 0, 33, 0, 0, 97, 0, 0, 0, 91, 0, 0, 0, 0, 0, 0, 0, 65, 63, 0, 0, 0, 0, 12, 0, 0, 31, 0, 0, 0, 0, 0, 59, 0, 0, 50, 0, 0, 0, 0, 26, 42, 0, 0, 52, 99, 0, 13, 80, 0, 0, 0, 0, 0, 0, 77, 0, 88, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 4, 0, 0, 0, 45, 0, 0, 50, 0, 0, 14, 0, 0, 0, 0, 0, 0, 98, 0, 19, 0, 0, 0, 0, 0, 0, 55, 95, 0, 57, 0, 0, 0, 0, 0, 0, 0, 0, 73, 0, 0, 69, 0, 0, 8, 0, 0, 0, 7, 13, 87, 0, 79, 0, 0, 0, 51, 0, 0, 58, 0, 0, 0, 0, 0, 0, 61, 0, 12, 0, 0, 0, 46, 0, 0, 0, 0, 0, 66, 0, 0, 0, 0], [0, 0, 0, 0, 66, 0, 68, 0, 0, 0, 0, 0, 0, 61, 0, 0, 12, 0, 0, 0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 47, 36, 0, 0, 93, 0, 0, 0, 32, 0, 0, 51, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 4, 10, 11, 0, 0, 0, 0, 0, 0, 49, 0, 0, 0, 33, 0, 0, 0, 0, 62, 0, 0, 59, 0, 48, 0, 0, 0, 0, 0, 0, 99, 0, 6, 0, 72, 5, 0, 0, 19, 0], [0, 13, 0, 0, 0, 80, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 92, 67, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 9, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 52, 0, 0, 0, 0, 38, 0, 0, 0, 57, 0, 55, 0, 0, 0, 0, 0, 0, 0, 0, 73, 86, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 56, 0, 0, 49, 0], [0, 68, 0, 0, 0, 0, 70, 0, 0, 16, 0, 0, 0, 0, 0, 96, 0, 0, 22, 0, 69, 71, 82, 0, 37, 0, 24, 86, 0, 0, 0, 0, 0, 0, 40, 36, 0, 0, 0, 13, 0, 0, 0, 0, 85, 0, 0, 0, 53, 88, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 56, 0, 0, 0, 0, 74, 92, 28, 0, 0, 0, 0, 0, 0, 0, 97, 0, 14, 0, 0, 0, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 100, 0], [0, 0, 40, 36, 0, 0, 0, 15, 0, 0, 0, 82, 0, 41, 0, 0, 71, 0, 0, 0, 11, 0, 54, 0, 0, 67, 0, 0, 0, 0, 0, 0, 23, 0, 28, 0, 0, 0, 0, 0, 0, 66, 0, 90, 0, 0, 91, 31, 0, 0, 0, 63, 0, 0, 0, 22, 0, 61, 0, 42, 14, 0, 84, 25, 0, 0, 0, 0, 83, 0, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 0, 80, 79, 0, 0, 0, 3, 0, 0, 58, 0, 18, 0, 0, 0, 0, 0, 93], [0, 23, 28, 0, 34, 0, 33, 89, 0, 75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 83, 57, 0, 50, 0, 0, 25, 0, 0, 0, 0, 0, 0, 0, 5, 20, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 37, 0, 0, 73, 41, 0, 69, 0, 0, 0, 0, 39, 0, 87, 0, 0, 0, 0, 0, 21, 0, 51, 0, 0, 0, 85, 76, 0, 58, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 91, 0, 0, 0, 0, 0, 0], [0, 73, 0, 41, 82, 0, 0, 0, 69, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 75, 0, 0, 0, 0, 0, 0, 0, 97, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42, 63, 0, 0, 0, 0, 0, 0, 9, 65, 0, 0, 0, 0, 0, 0, 0, 0, 90, 0, 98, 20, 0, 5, 0, 0, 0, 0, 0, 0, 84, 50, 95, 57, 0, 48, 0, 0, 0, 0, 0, 0, 58, 0, 0, 85, 0, 4, 76, 0, 0, 0, 0, 77, 0, 3, 0, 0, 0, 0], [95, 30, 0, 0, 0, 0, 0, 57, 14, 0, 0, 2, 0, 52, 0, 0, 99, 0, 0, 0, 18, 0, 0, 88, 0, 0, 0, 0, 0, 58, 0, 64, 7, 0, 0, 0, 21, 0, 0, 3, 0, 0, 54, 10, 0, 0, 0, 92, 0, 0, 0, 97, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 70, 0, 0, 35, 31, 96, 0, 0, 0, 0, 0, 26, 0, 63, 65, 55, 43, 0, 0, 47, 0, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 0, 0, 0, 0, 21, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 0, 70, 0, 68, 0, 0, 61, 0, 0, 0, 0, 0, 63, 42, 26, 0, 0, 0, 0, 0, 50, 14, 0, 0, 0, 100, 5, 2, 0, 0, 0, 0, 99, 0, 0, 0, 0, 40, 17, 27, 0, 43, 0, 0, 15, 0, 0, 32, 0, 73, 0, 0, 69, 0, 41, 0, 0, 97, 0, 0, 75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 0, 92, 0], [0, 0, 0, 0, 54, 0, 49, 0, 11, 1, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 5, 0, 0, 0, 0, 84, 0, 0, 0, 0, 78, 0, 81, 0, 0, 0, 0, 0, 37, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 93, 0, 85, 0, 0, 76, 0, 0, 0, 0, 39, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 16, 45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 96], [0, 0, 0, 0, 76, 58, 0, 4, 0, 0, 0, 3, 7, 0, 39, 8, 0, 0, 0, 0, 0, 0, 26, 0, 0, 42, 0, 0, 0, 0, 0, 16, 0, 0, 90, 0, 0, 66, 0, 0, 20, 5, 0, 0, 0, 0, 0, 100, 0, 0, 57, 25, 0, 0, 0, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 0, 54, 0, 0, 97, 0, 0, 33, 0, 0, 0, 0, 28], [0, 0, 0, 0, 0, 63, 0, 38, 46, 0, 0, 0, 0, 0, 0, 90, 0, 0, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 86, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 79, 0, 21, 0, 0, 0, 0, 0, 53, 88, 51, 0, 85, 93, 0, 0, 0, 28, 97, 0, 0, 0, 0, 62, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 100, 0, 0, 83, 0, 95, 0, 50, 0, 30, 48, 0, 0], [0, 0, 0, 0, 2, 0, 19, 0, 0, 99, 0, 78, 0, 0, 0, 0, 0, 0, 50, 0, 77, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 75, 0, 0, 0, 0, 62, 0, 56, 0, 60, 0, 0, 0, 0, 0, 67, 0, 0, 0, 63, 61, 0, 0, 0, 65, 0, 0, 0, 0, 31, 0, 0, 0, 91, 66, 35, 69, 0, 0, 0, 0, 71, 0, 0, 82, 0, 0, 0, 0, 0, 0, 13, 0, 27, 0, 0], [0, 0, 0, 37, 0, 0, 85, 0, 0, 0, 80, 0, 0, 15, 7, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 61, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 0, 6, 0, 22, 0, 0, 0, 0, 0, 83, 0, 0, 0, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 57, 0, 0, 0, 0, 0, 0, 0, 74, 10, 0, 0, 0, 0, 0, 0, 0, 0, 79, 0, 0], [0, 0, 13, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 53, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 67, 0, 0, 33, 0, 0, 34, 0, 0, 0, 9, 0, 30, 0, 0, 0, 0, 0, 0, 0, 45, 0, 52, 0, 0, 42, 0, 99, 0, 0, 26, 72, 0, 43, 0, 55, 0, 0, 95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 86, 5, 69, 0, 0, 0, 0, 0, 75, 79, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 44, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41, 0, 71, 0, 0, 69, 0, 0, 95, 0, 0, 0, 36, 0, 0, 0, 0, 0, 93, 82, 0, 0, 0, 88, 0, 0, 0, 77, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 76, 0, 51, 0, 44, 0, 0, 0, 0, 0, 0, 62, 39, 79, 0, 87, 0, 0, 0, 0, 0, 0, 0, 70, 0, 0, 0, 0, 0, 98, 0, 63, 0, 0, 0, 0, 52, 0], [0, 0, 2, 100, 0, 69, 71, 0, 0, 19, 36, 0, 0, 57, 0, 78, 0, 0, 27, 55, 0, 0, 28, 0, 0, 23, 75, 0, 0, 94, 0, 51, 0, 0, 76, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 98, 0, 0, 0, 0, 38, 0, 0, 0, 30, 83, 45, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 80, 13], [0, 0, 3, 21, 28, 0, 0, 62, 0, 0, 0, 0, 0, 4, 0, 76, 0, 0, 1, 0, 0, 0, 84, 31, 83, 0, 0, 59, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 95, 0, 41, 69, 0, 20, 0, 0, 0, 0, 0, 73, 17, 0, 0, 0, 47, 64, 0, 0, 0, 0, 0, 0, 0, 88, 0, 24, 93, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 96, 34, 0, 0, 0, 56, 16, 0, 0, 0, 35, 0], [0, 0, 0, 0, 0, 11, 1, 0, 0, 0, 0, 28, 0, 0, 23, 0, 0, 0, 75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 2, 19, 0, 0, 0, 0, 0, 36, 55, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 64, 0, 80, 0, 47, 0, 17, 77, 0, 0, 0, 0, 92, 16, 49, 68, 35, 0, 0, 0, 46, 0, 0, 0, 96, 12, 0, 0, 0], [9, 0, 0, 0, 96, 0, 0, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 57, 0, 0, 27, 0, 95, 0, 5, 0, 0, 71, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 47, 0, 0, 0, 0, 15, 0, 0, 93, 0, 0, 0, 81, 0, 0, 0, 87, 23, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 52, 0, 0, 0, 0, 0, 66, 0, 84, 45, 70, 0, 0], [0, 0, 0, 0, 0, 98, 0, 0, 0, 0, 83, 0, 45, 0, 0, 0, 0, 0, 48, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 82, 0, 0, 0, 53, 0, 0, 0, 0, 0, 0, 0, 87, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 34, 0, 0, 0, 0, 0, 0, 0, 54, 16, 68, 0, 67, 49, 90, 56, 0, 92, 0, 20, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 43, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 40, 55, 0, 0, 25, 0, 0, 0, 0, 0, 73, 2, 0, 0, 0, 0, 58, 0, 0, 0, 74, 60, 1, 0, 0, 0, 0, 0, 39, 0, 0, 0, 0, 0, 23, 28, 0, 0, 90, 0, 0, 0, 56, 0, 0, 0, 65, 0, 96, 12, 0, 0, 0, 0, 0, 0, 66, 0, 0, 0, 0, 0, 0, 0, 31, 0, 0, 99, 0, 52, 0, 0, 0, 0, 98, 38, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 81, 0, 0, 0, 0, 0, 82], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 52, 6, 42, 0, 72, 0, 0, 63, 0, 0, 0, 24, 0, 0, 0, 53, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 11, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 79, 0, 0, 0, 68, 54, 0, 49, 16, 0, 0, 0, 0, 34, 0, 9, 0, 0, 33, 0, 0, 0, 0, 0, 95, 0, 0, 0, 0, 0, 36, 0, 78, 0, 0, 73, 0, 71, 0, 20, 0, 0, 0]]

    logger.info(f"Puzzle: {board_puzzle}")

    start_msg = f"Starting exploration of puzzle"
    logger.info(start_msg)
    alert_sender.send(start_msg)

    start_time = time.time()

    explorer = SolvableSamplesExplorer(board_puzzle)
    board_solved = explorer.explore(AlgorithmType.CSP_ITERATIVE_MULTIPROCESS)

    end_time = time.time()
    elapsed_time = end_time - start_time

    logger.info(f"Solution found in {elapsed_time} seconds\n{board_solved}\n")
    alert_sender.send(f"Solution found in {elapsed_time} seconds")

    if board_solved is not None:
        explorer.write_report(board_puzzle, is_dev=True)


if __name__ == '__main__':
    main()
