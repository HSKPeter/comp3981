        
        document.getElementById("solve-brute-force").addEventListener("click", function () {
            window.location.href = "solved_screen.html";
        });
        document.getElementById("solve-csp").addEventListener("click", function () {
            window.location.href = "solved_screen.html";
        });
        document.getElementById("clear").addEventListener("click", function () {
            window.location.href = "main_menu.html";
        });


const size = localStorage.getItem("size");
const board_values = JSON.parse(localStorage.getItem("board"));

document.getElementById("sudoku_container").style.gridTemplateColumns = "repeat(" + size + ", " + 10 + "px)";
document.getElementById("sudoku_container").style.gridTemplateRows = "repeat(" + size + ", " + 10 + "px)";

var sudoku_container = document.getElementById('sudoku_container');


console.log(board_values)
console.log(typeof board_values)

for (let i = 0; i < size; i++){
    for (let j = 0; j < size; j++) {
        var cell_to_insert = document.createElement('div');
        cell_to_insert.setAttribute('id', 'cell ' + i + "-" + j)
        if (board_values[i][j] == 0) cell_to_insert.innerHTML = '';
        else cell_to_insert.innerHTML = board_values[i][j]

        sudoku_container.appendChild(cell_to_insert);
    }
    
}   

if (size > 25) {
    document.getElementById("solve-brute-force").remove()
}


