
import SudokuPuzzleGenerator from "./sudoku_generator.js";

document.getElementById("solve-brute-force").addEventListener("click", function () {
    window.location.href = "solved_screen.html";
});
document.getElementById("solve-csp").addEventListener("click", function () {
    window.location.href = "solved_screen.html";
});
document.getElementById("clear").addEventListener("click", function () {
    window.location.href = "main_menu.html";
});

const urlParams = new URLSearchParams(window.location.search);
const source = urlParams.get('source');
const size = localStorage.getItem("size");
const sudoku_container = document.getElementById('sudoku_container');

async function main() {
    let board_values;

    if (size > 25) {
        document.getElementById("solve-brute-force").remove()
    }
    
    
    if (source === "file") {
        board_values = JSON.parse(localStorage.getItem("board"));
    }
    
    if (source === "generate") {
        const generator = new SudokuPuzzleGenerator(+size)
        board_values = await generator.generateNewPuzzle()
        generator.printBoard()
    }
    
    
    sudoku_container.style.gridTemplateColumns = "repeat(" + size + ", " + 10 + "px)";
    sudoku_container.style.gridTemplateRows = "repeat(" + size + ", " + 10 + "px)";
    
    
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            var cell_to_insert = document.createElement('div');
            cell_to_insert.setAttribute('id', 'cell ' + i + "-" + j)
            if (board_values[i][j] == 0) cell_to_insert.innerHTML = '';
            else cell_to_insert.innerHTML = board_values[i][j]
    
            sudoku_container.appendChild(cell_to_insert);
        }
    }
}

main()

