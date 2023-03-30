from algo import *
import os
import copy
from puzzle_loader import PuzzleLoader
from multiprocessing import Process


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


def main():
    board_size = 9

    board_puzzle_loader = PuzzleLoader()
    board_puzzle = board_puzzle_loader.load_unsolved_puzzle(size=board_size)
    explorer = SolvableSamplesExplorer(board_puzzle)
    board_solved = explorer.explore(AlgorithmType.CSP_ITERATIVE)

    print(board_puzzle)
    print(board_solved)

    if board_solved is not None:
        explorer.write_report(board_puzzle, is_dev=False)


if __name__ == '__main__':
    timeout_seconds = 60 * 10

    p1 = Process(target=main, name='explore solvable puzzle (1)')
    # p2 = Process(target=main, name='explore solvable puzzle (2)')

    p1.start()
    # p2.start()

    p1.join(timeout=timeout_seconds)
    # p2.join(timeout=timeout_seconds)

    p1.terminate()
    # p2.terminate()

    if p1.exitcode is None:
        print(f'{p1} timeouts after {timeout_seconds}s!')

    # if p2.exitcode is None:
    #     print(f'{p2} timeouts after {timeout_seconds}s!')
