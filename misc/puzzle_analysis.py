import copy

from algo_util import get_sub_square_index
from puzzle_loader import PuzzleLoader


class PuzzleAnalyzer:
    def __init__(self, puzzle):
        self.puzzle = copy.deepcopy(puzzle)
        self.n = len(self.puzzle)
        self.values = {(row, col, get_sub_square_index(self.n, row, col)): puzzle[row][col] for row in range(self.n) for col in range(self.n)}
        self.every_cell_neighbour = self.find_every_cell_neighbours()

    def count_empty_cells(self):
        empty_cell_count = 0
        for row in self.puzzle:
            for cell in row:
                if cell == 0:
                    empty_cell_count += 1
        return empty_cell_count

    def compute_empty_cell_ratio(self):
        return self.count_empty_cells() / (self.n * self.n)

    def count_domain_size(self, cell: (int, int, int)):
        if self.values[cell] != 0:
            return 1

        domain = set(range(1, self.n + 1))

        cell_neighbours = self.every_cell_neighbour[cell]
        for neighbour in cell_neighbours:
            if self.values[neighbour] in domain:
                domain.remove(self.values[neighbour])

        return len(domain)

    def compute_metrics_of_difficulty_level(self):
        total_domain_size = 0
        for cell in self.values.keys():
            total_domain_size += self.count_domain_size(cell)

        return total_domain_size

    def find_every_cell_neighbours(self):
        return {cell: self.find_cell_neighbours(cell) for cell in self.values.keys()}

    def find_cell_neighbours(self, cell: (int, int, int)):
        cell_neighbours = set()
        row_index, col_index, sub_square_index = cell
        for counter_cell in self.values.keys():
            counter_cell_row_index, counter_cell_col_index, counter_cell_sub_square_index = counter_cell
            is_same_row = row_index == counter_cell_row_index
            is_same_col = col_index == counter_cell_col_index
            is_same_sub_square = sub_square_index == counter_cell_sub_square_index
            is_different_cell = cell != counter_cell

            if is_different_cell and (is_same_row or is_same_col or is_same_sub_square):
                cell_neighbours.add(counter_cell)

        return cell_neighbours


def main():
    puzzle_loader = PuzzleLoader()

    for i in range(10):
        mega_sudoku_puzzle = puzzle_loader.load_random_unsolved_puzzle(9)
        puzzle_analyzer = PuzzleAnalyzer(mega_sudoku_puzzle)
        print(f"Random 100x100 puzzle is: {mega_sudoku_puzzle}")
        print(f"Empty cell ratio: {puzzle_analyzer.compute_empty_cell_ratio()}")
        print(f"Difficulty level: {puzzle_analyzer.compute_metrics_of_difficulty_level()}\n")



if __name__ == "__main__":
    main()