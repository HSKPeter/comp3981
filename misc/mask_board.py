import math
from algo_util import get_sub_square_index
import random


class BoardMaskManager:
    def __init__(self, board,
                 required_empty_cell_ratio=0.75,
                 required_count_of_filled_cells_in_row=1,
                 required_count_of_filled_cells_in_col=1,
                 required_count_of_filled_cells_in_sub_square_row=1,
                 required_count_of_filled_cells_in_sub_square_col=1):
        self.board = board
        self.board_size = len(board)

        sqrt_board_size = int(math.sqrt(self.board_size))
        self.sub_square_size = sqrt_board_size
        self.required_count_of_filled_cells_in_row = required_count_of_filled_cells_in_row
        self.required_count_of_filled_cells_in_col = required_count_of_filled_cells_in_col

        self.required_count_of_filled_cells_in_sub_square_row = required_count_of_filled_cells_in_sub_square_row
        self.required_count_of_filled_cells_in_sub_square_col = required_count_of_filled_cells_in_sub_square_col

        self.required_empty_cell_ratio = required_empty_cell_ratio
        self.total_cell_count = self.board_size * self.board_size
        self.required_empty_cell_count = math.ceil(self.total_cell_count * required_empty_cell_ratio)
        self.filled_cells_count_cap = self.total_cell_count - self.required_empty_cell_count

    def produce_random_board_mask(self):
        sub_square_masks = []
        for i in range(self.board_size):
            sub_square_masks.append(self.get_good_sub_square_mask())

        empty_cell_count = 0
        sub_square_size = int(math.sqrt(self.board_size))

        board_mask = [[True for _ in range(self.board_size)] for _ in range(self.board_size)]
        filled_cells = set()

        for i in range(self.board_size):
            for j in range(self.board_size):
                sub_square_index = get_sub_square_index(self.board_size, i, j)
                sub_square_mask = sub_square_masks[sub_square_index]
                board_mask[i][j] = sub_square_mask[i % sub_square_size][j % sub_square_size]
                if board_mask[i][j] is False:
                    empty_cell_count += 1
                else:
                    filled_cells.add((i, j))

        if empty_cell_count >= self.required_empty_cell_count:
            return board_mask

        additional_empty_cell_count_needed = self.required_empty_cell_count - empty_cell_count

        for i in range(additional_empty_cell_count_needed):
            random_cell = random.choice(list(filled_cells))
            board_mask[random_cell[0]][random_cell[1]] = False
            filled_cells.remove(random_cell)

        return board_mask

    def produce_good_board_mask(self):
        board_mask = self.produce_random_board_mask()

        while self.is_good_board_mask(board_mask) is False:
            board_mask = self.produce_random_board_mask()

        return board_mask

    def is_good_board_mask(self, board_mask):
        for row in board_mask:
            if row.count(True) < self.required_count_of_filled_cells_in_row:
                return False

        for col in range(self.board_size):
            col_values = [row[col] for row in self.board]
            if col_values.count(True) < self.required_count_of_filled_cells_in_col:
                return False

        return True

    def get_good_sub_square_mask(self):
        sub_square_mask = self.get_random_sub_square_mask()

        while self.is_good_sub_square_mask(sub_square_mask) is False:
            sub_square_mask = self.get_random_sub_square_mask()

        return sub_square_mask

    def get_random_sub_square_mask(self):
        """
        Get sub square mask for a given size, which is a 2D array of random boolean values
        """
        count_of_filled_cells = 0

        sub_square_mask = [[] for _ in range(self.sub_square_size)]
        for i in range(self.sub_square_size):
            for j in range(self.sub_square_size):
                boolean_choice = random.random() > 0.5 if count_of_filled_cells < self.filled_cells_count_cap else False
                if boolean_choice is True:
                    count_of_filled_cells += 1
                sub_square_mask[i].append(boolean_choice)

        return sub_square_mask

    def is_good_sub_square_mask(self, sub_square_mask):
        """
        Check if the given sub square mask has enough filled cells in each row and col to satisfy the required count
        """

        for row in sub_square_mask:
            if row.count(True) < self.required_count_of_filled_cells_in_sub_square_row:
                return False

        for col in range(self.sub_square_size):
            col_values = [row[col] for row in sub_square_mask]
            if col_values.count(True) < self.required_count_of_filled_cells_in_sub_square_col:
                return False

        return True

    def mask_board(self):
        board_mask = self.produce_good_board_mask()

        masked_board = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]

        for i in range(self.board_size):
            for j in range(self.board_size):
                if board_mask[i][j] is True:
                    masked_board[i][j] = self.board[i][j]

        return masked_board


def main():
    sample_board = [[7, 6, 2, 4, 9, 8, 5, 1, 3], [9, 3, 1, 2, 5, 6, 4, 8, 7], [4, 5, 8, 1, 3, 7, 2, 6, 9], [5, 1, 9, 7, 8, 2, 3, 4, 6], [6, 8, 7, 9, 4, 3, 1, 5, 2], [3, 2, 4, 5, 6, 1, 9, 7, 8], [2, 9, 6, 8, 1, 4, 7, 3, 5], [8, 4, 5, 3, 7, 9, 6, 2, 1], [1, 7, 3, 6, 2, 5, 8, 9, 4]]

    board_mask_manager = BoardMaskManager(sample_board)

    print(f"Original board is: {sample_board}\n")
    print(f"Masked board is: {board_mask_manager.mask_board()}")


if __name__ == "__main__":
    main()