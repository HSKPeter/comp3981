class SudokuPuzzleGenerator {
    constructor(N) {
        this.board = Array.from(Array(N), () => new Array(N).fill(0));
        this.solver = new SudokuValidityChecker(this.board);
    }

    generateNewSolution() {
        this.clearBoard()
        const random = (min, max) => Math.floor(Math.random() * (max - min)) + min;
        const N = this.board.length
        let col = 0
        let row = 0
        let filledValues = 0
        while (filledValues < N * N * 0.25) {
            col++;
            if (col >= N) {
                col = 0
                row++
                if (row >= N) {
                    col = 0
                    row = 0
                }
            }
            if (Math.random() < 0.25) {
                let newValue = random(1, N + 1)
                if (this.solver.isValidValue(row, col, newValue)) {
                    this.updateBoard(row, col, newValue)
                    this.printBoard()
                    filledValues++
                }
            }
        }
    }



    updateBoard(row, col, value) {
        this.board[row][col] = value; // update the board with the new value
        this.solver.updateSets(row, col, value)
    }

    clearBoard() {
        const N = this.board.length;
        this.board = Array.from(Array(N), () => new Array(N).fill(0));
        this.solver = new SudokuValidityChecker(this.board);
    }

    printBoard() {
        const boardString = this.board.map(row => row.join(' ')).join('\n');
        console.log("\n" + boardString + "\n");
    }
}


class SudokuValidityChecker {
    constructor(board) {
        this.rowSets = this.generateRowSets(board);
        this.colSets = this.generateColSets(board);
        this.squareSets = this.generateSquareSets(board);
        this.N = board.length
    }

    generateRowSets(board) {
        return board.map(row => new Set(row));
    }

    generateColSets(board) {
        const colSets = [];
        for (let col = 0; col < board.length; col++) {
            const colSet = new Set();
            for (let row = 0; row < board.length; row++) {
                colSet.add(board[row][col]);
            }
            colSets.push(colSet);
        }
        return colSets;
    }

    generateSquareSets(board) {
        const squareSize = Math.sqrt(board.length);
        const squareSets = [];
        for (let i = 0; i < board.length; i += squareSize) {
            for (let j = 0; j < board.length; j += squareSize) {
                const squareSet = new Set();
                for (let row = i; row < i + squareSize; row++) {
                    for (let col = j; col < j + squareSize; col++) {
                        squareSet.add(board[row][col]);
                    }
                }
                squareSets.push(squareSet);
            }
        }
        return squareSets;
    }

    updateSets(row, col, newValue) {
        // Update the rowSet
        this.rowSets[row].add(newValue);

        // Update the colSet
        this.colSets[col].add(newValue);

        // Update the squareSet
        const squareIndex = this.getSquareIndex(row, col);
        this.squareSets[squareIndex].add(newValue);
    }

    isValidValue(row, col, value) {
        // Check if value is in the rowSet
        if (this.rowSets[row].has(value)) {
            return false;
        }

        // Check if value is in the colSet
        if (this.colSets[col].has(value)) {
            return false;
        }

        // Check if value is in the squareSet
        const squareIndex = this.getSquareIndex(row, col);
        if (this.squareSets[squareIndex].has(value)) {
            return false;
        }

        return true;
    }

    getSquareIndex(rowIndex, columnIndex) {
        const squareSize = Math.sqrt(this.N);
        const row = Math.floor(rowIndex / squareSize);
        const col = Math.floor(columnIndex / squareSize);
        return row * squareSize + col;
    }
}




const sudokuGenerator = new SudokuPuzzleGenerator(100); // create a 9x9 Sudoku board
console.log(sudokuGenerator.solver.squareSets)
sudokuGenerator.printBoard()
sudokuGenerator.generateNewSolution()
sudokuGenerator.printBoard()
