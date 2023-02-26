
        document.getElementById("solve-csp").addEventListener("click", function () {
            window.location.href = "solved_screen.html";
        });
        document.getElementById("clear").addEventListener("click", function () {
            window.location.href = "main_menu.html";
        });

        let size = localStorage.getItem("size");

document.getElementById("sudoku_container").style.gridTemplateColumns = "repeat(" + size + ", " + 270/size + "px)";
document.getElementById("sudoku_container").style.gridTemplateRows = "repeat(" + size + ", " + 270/size + "px)";

var sodoku_container = document.getElementById('sudoku_container');

for (let i = 0; i < size * size; i++){
    var cell_to_insert = document.createElement('div');
    cell_to_insert.setAttribute('id', 'cell ' + i)
    cell_to_insert.innerHTML = '';
    sodoku_container.appendChild(cell_to_insert);
}   