import json

class PuzzleLoader:
    def load(self, size):

        # TODO return different 2D arrays subject to the size argument

        # TODO Mask the value in the solved sudoku board
        with open('assets/solved_sudoku/sudoku9x9/9x9_sample_0.txt', 'r') as file:
            contents = file.read()
            return json.loads(contents)