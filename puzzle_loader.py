import json
import random
class PuzzleLoader:
    def load(self, size):
        """
        Loads a random sudoku puzzle of the correct size
        """
        sample = random.randint(0, 9)
        with open('assets/solved_sudoku/sudoku{0}x{0}/{0}x{0}_sample_{1}.txt'.format(size, sample), 'r') as file:
            contents = file.read()
            board = json.loads(contents)
            board = self.mask_puzzle(board)
            return board
    
    def mask_puzzle(self, board):
        """
        randomly sets 75% of a board to 0s and returns the board
        """
        side = len(board)
        size = side*side
        values_to_mask = int(size * 0.75)
        masked_values = 0
        col = 0
        row = 0
        while (masked_values <= values_to_mask): 
            if random.random() < 0.5:
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
    print(loader.load(12))

if __name__ == '__main__':
    main()

        
