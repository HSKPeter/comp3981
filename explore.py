from algo import *
import os
import copy
from puzzle_loader import PuzzleLoader
from multiprocessing import Process
import logging
from datetime import datetime
import uuid


def config_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    now = datetime.now()
    formatted_date_time = now.strftime('%Y-%m-%d_%H%M')

    package_directory = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(package_directory, "log", f"{formatted_date_time}_exploration_{uuid.uuid4().hex}.log")

    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.flush = file_handler.stream.flush
    formatter = logging.Formatter('%(asctime)s - %(levelname)s\n %(message)s\n')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger


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


def explore_solvable_board():
    board_size = 100

    logger = config_logger()
    logger.info('Start exploring solvable puzzle')
    logger.handlers[0].flush()

    board_puzzle_loader = PuzzleLoader()
    board_puzzle = board_puzzle_loader.load_unsolved_puzzle(size=board_size)
    explorer = SolvableSamplesExplorer(board_puzzle)
    board_solved = explorer.explore(AlgorithmType.CSP_ITERATIVE)

    logger.info(board_puzzle)
    logger.info(board_solved)

    if board_solved is not None:
        explorer.write_report(board_puzzle, is_dev=False)


if __name__ == '__main__':
    timeout_seconds = 60 * 10

    p = Process(target=explore_solvable_board, name='explore solvable puzzle (1)')
    p.start()
    p.join(timeout=timeout_seconds)

    p.terminate()

    if p.exitcode is None:
        print(f'{p} timeouts after {timeout_seconds}s!')
