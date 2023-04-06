import os

from misc.algo import AlgorithmType
from benchmark import BenchmarkTestRunner
from misc.puzzle_analysis import PuzzleAnalyzer
from puzzle_loader import PuzzleLoader


class BoardSampleManager:
    _package_directory = os.path.dirname(os.path.abspath(__file__))
    _default_raw_samples_path = os.path.join(_package_directory, "assets", "raw_samples")

    @classmethod
    def save_board(cls, file_path, filename, board):
        result = ""

        for row in board:
            row_str = ",".join(str(num) for num in row)
            result += row_str + "\n"

        cls.write_file(file_path, filename, str.strip(result))

    @classmethod
    def read_samples_file(cls, filename):
        with open(f"{cls._default_raw_samples_path}/{filename}", "r") as file:
            return file.read()

    @classmethod
    def write_file(cls, file_path, filename, content):
        with open(os.path.join(cls._package_directory, f"{file_path}/{filename}"), "w") as file:
            file.write(content)

    @classmethod
    def save_easy_standard_samples(cls):
        file_path = "assets/standard_samples/9x9/easy"
        easy_samples_text = cls.read_samples_file("list-easy-samples-9x9.txt")
        easy_samples = easy_samples_text.split("========")
        for i in range(len(easy_samples)):
            easy_sample = easy_samples[i]
            rows = [line for line in easy_sample.split("\n") if line != '']

            board_in_chars = [list(row) for row in rows]
            board = [[int(num) for num in row] for row in board_in_chars]

            cls.save_board(file_path, f"easy_sample_{str(i + 1).zfill(2)}.txt", board)

        print(f"Saved {len(easy_samples)} easy samples to {file_path}")

    @classmethod
    def save_hard_standard_samples(cls):
        file_path = "assets/standard_samples/9x9/hard"
        hard_samples_text = cls.read_samples_file("list-hard-samples-9x9.txt")
        hard_samples = [line for line in hard_samples_text.split("\n") if line != '']
        for i in range(len(hard_samples)):
            hard_sample = hard_samples[i]
            hard_sample_with_empty_cell_as_zero = hard_sample.replace(".", "0")
            int_list = [int(num) for num in hard_sample_with_empty_cell_as_zero]
            board = [int_list[j:j + 9] for j in range(0, len(int_list), 9)]
            cls.save_board(file_path, f"hard_sample_{str(i + 1).zfill(2)}.txt", board)

        print(f"Saved {len(hard_samples)} hard samples to {file_path}")

    @classmethod
    def save_custom_samples(cls, samples, size):
        file_path = os.path.join("assets", f"solvable_puzzle/{size}x{size}")
        for i in range(len(samples)):
            sample = samples[i]
            cls.save_board(file_path=os.path.join("assets", f"solvable_puzzle/{size}x{size}"),
                           filename=f"{size}x{size}_sample_{str(i + 1).zfill(2)}.txt",
                           board=sample)

        print(f"Saved {len(samples)} custom samples to {file_path}")

    @classmethod
    def save_custom_sample(cls, sample, size, index):
        file_path = os.path.join("assets", f"solvable_puzzle/{size}x{size}_pool")
        filename = f"{size}x{size}_sample_{str(index + 1).zfill(2)}.txt"
        cls.save_board(file_path=file_path,
                       filename=filename,
                       board=sample)

        print(f"Saved sample to {file_path}/{filename}")

    @staticmethod
    def board_to_standard_text_format(board):
        num_strings_having_int_with_comma_separated = []
        for row in board:
            num_strings_having_int_with_comma_separated.append(",".join([str(i) for i in row]))

        return "\n".join(num_strings_having_int_with_comma_separated)


def main():
    benchmark_test_runner = BenchmarkTestRunner()
    puzzle_loader = PuzzleLoader()

    board_sizes = [9,12,16,25]
    puzzles_count_for_each_size =  10

    for board_size in board_sizes:
        for i in range(puzzles_count_for_each_size):
            if board_size == 100 and i >= 1:
                break

            board_puzzle = puzzle_loader.load_random_unsolved_puzzle(board_size)
            puzzle_analyzer = PuzzleAnalyzer(board_puzzle)
            empty_cell_ratio = puzzle_analyzer.compute_empty_cell_ratio()

            board_puzzle_in_standard_format = BoardSampleManager.board_to_standard_text_format(board_puzzle)
            print(f"Loaded random puzzle:\n{board_puzzle_in_standard_format}")
            print(f"Empty cell ratio: {empty_cell_ratio}\n\n")

            if board_size < 16:
                benchmark_test_runner.run_benchmark(board_puzzle, AlgorithmType.CSP_RECURSIVE)
            elif board_size <= 25:
                benchmark_test_runner.run_benchmark(board_puzzle, AlgorithmType.CSP_ITERATIVE_MULTIPROCESS)

            if empty_cell_ratio > 0.75:
                BoardSampleManager.save_custom_sample(board_puzzle, board_size, index=i)


if __name__ == "__main__":
    main()