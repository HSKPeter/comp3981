import copy
import json
import os
import random


class PuzzleLoader:

    _package_directory = os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def read_file(filename):
        with open(filename, 'r') as file:
            return file.read()
        
    @staticmethod
    def standard_text_format_to_2d_array(text):
        rows = text.split("\n")
        result = []
        for row in rows:
            nums = [int(num) for num in row.split(",")]
            result.append(nums)
        return result

    def load_unsolved_9x9_puzzle_from_standard_samples(self, is_easy=True):
        """
        Loads a random sudoku puzzle of the correct size
        """
        sample_index = random.randint(1, 50)
        difficulty_level = "easy" if is_easy else "hard"
        file_path = f"assets/standard_samples/9x9/{difficulty_level}/{difficulty_level}_sample_{str(sample_index).zfill(2)}.txt"
        file_text_content = self.read_file(file_path)
        return self.standard_text_format_to_2d_array(file_text_content)

    def load_txt_file_in_standard_format(self, size, sample_index=None, is_easy=False):
        """
        Loads a random sudoku puzzle of the correct size
        """
        if size == 9:
            if sample_index is None:
                sample_index = random.randint(1, 50) if is_easy else random.randint(1, 95)
            difficulty = "easy" if is_easy else "hard"
            file_path = os.path.join("assets", "standard_samples", "9x9", difficulty, f"{difficulty}_sample_{str(sample_index).zfill(2)}.txt")
            file_content = self.read_file(os.path.join(self._package_directory, file_path))
            return file_path, file_content
        else:
            if sample_index is None:
                sample_index = random.randint(1, 4)
            file_path = os.path.join("assets", "solvable_puzzle", f"{size}x{size}", f"{size}x{size}_sample_{str(sample_index).zfill(2)}.txt")
            file_content = self.read_file(os.path.join(self._package_directory, file_path))
            return file_path, file_content

    def load_from_2d_array_txt_file(self, size):
        """
        Loads a random sudoku puzzle of the correct size
        """
        sample = random.randint(0, 9)
        file_path = 'assets/solved_sudoku/sudoku{0}x{0}/{0}x{0}_sample_{1}.txt'.format(size, sample)
        file_text_content = self.read_file(file_path)
        return json.loads(file_text_content)

    def load_predefined_puzzle(self, size):
        sample = random.randint(1, 10)
        file_path = 'assets/solved_sudoku/sudoku{0}x{0}/{0}x{0}_sample_{1}.txt'.format(size, sample)
        file_text_content = self.read_file(file_path)
        return self.standard_text_format_to_2d_array(file_text_content)

    def load_random_unsolved_puzzle(self, size):
        solved_board_sample = self.load_from_2d_array_txt_file(size)
        masked_board = self.mask_puzzle(solved_board_sample)
        return masked_board

    def load_unsolved_puzzle(self, size, sample_index, is_easy=False):
        file_path, file_content = self.load_txt_file_in_standard_format(size, sample_index, is_easy)
        rows = file_content.split("\n")
        result = []
        for row in rows:
            nums = [int(num) for num in row.split(",")]
            result.append(nums)
        return file_path, file_content, result

    @staticmethod
    def mask_puzzle(board):
        """
        randomly sets 75% of a board to 0s and returns the board
        """
        board = copy.deepcopy(board)
        side = len(board)
        size = side*side
        values_to_mask = int(size * 0.75)
        masked_values = 0
        col = 0
        row = 0
        while masked_values <= values_to_mask:
            if board[row][col] != 0 and random.random() < 0.5:
                board[row][col] = 0
                masked_values += 1
            col += 1
            if col >= side:
                col = 0
                row += 1
                if row >= side:
                    col = 0
                    row = 0
        return board


def main():
    loader = PuzzleLoader()
    print(loader.load_from_2d_array_txt_file(12))


if __name__ == '__main__':
    main()

        
