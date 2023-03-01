
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



const sudoku_container = document.getElementById('sudoku_container');
const size = localStorage.getItem("size");
let board_values = JSON.parse(localStorage.getItem("board"));
const urlParams = new URLSearchParams(window.location.search);
const source = urlParams.get('source');

sudoku_container.style.gridTemplateColumns = "repeat(" + size + ", " + 10 + "px)";
sudoku_container.style.gridTemplateRows = "repeat(" + size + ", " + 10 + "px)";

if (size > 25) {
    document.getElementById("solve-brute-force").remove()
}

if (source === "generate") {
    const generator = new SudokuPuzzleGenerator(+size)
    board_values = generator.generateNewPuzzle()
    generator.printBoard()
}

console.log(board_values)


for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
        var cell_to_insert = document.createElement('div');
        cell_to_insert.setAttribute('id', 'cell ' + i + "-" + j)
        if (board_values[i][j] == 0) cell_to_insert.innerHTML = '';
        else cell_to_insert.innerHTML = board_values[i][j]

        sudoku_container.appendChild(cell_to_insert);
    }

}


