from misc.algo import *
import copy
from puzzle_loader import PuzzleLoader
from misc.log_util import logger, log_format
from datetime import datetime
import uuid
import os
import time
from misc.alert_sender import AlertSender
from board_sample_manage import BoardSampleManager

now = datetime.now()
formatted_date_time = now.strftime('%Y-%m-%d_%H%M')

package_directory = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(package_directory, "log", f"{formatted_date_time}_exploration_{uuid.uuid4().hex}.log")
logger.add(log_file_path, format=log_format)


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
