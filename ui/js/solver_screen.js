
import SudokuPuzzleGenerator from "./sudoku_generator.js";

const timeContainer = document.getElementById("timeContainer");
const timeText = document.getElementById("time")
const title = document.getElementById("title")

let deltaTime;
var startTime;
let endTime;
let sum = 0;

document.getElementById("solve-brute-force").addEventListener("click", function () {
    displayTime(solveWithBruteForce);
});
document.getElementById("solve-csp").addEventListener("click", function () {
    displayTime(solveWithCSP);
});
document.getElementById("clear").addEventListener("click", function () {
    window.location.href = "main_menu.html";
});

// Replace contents of this function with the real algorithm
function solveWithBruteForce() {
    localStorage.setItem("Algorithm", "Brute Force")
    setTimeout(showTime, 2500);
}

function showTime() {

    endTime = Date.now();
    deltaTime = endTime - startTime;
    timeText.innerText = deltaTime;
    timeContainer.style.display = "block";
    let algorithm = localStorage.getItem("Algorithm");
    if (algorithm === "CSP") {
        title.innerHTML = "Solved with CSP"
    } else {
        title.innerHTML = "Solved with Brute Force"
    }

}

// Replace contents of this function with the real algorithm
function solveWithCSP() {
    localStorage.setItem("Algorithm", "CSP")
    setTimeout(showTime, 5000)
}

function displayTime(solveAlgorithm) {
    startTime = Date.now()
    solveAlgorithm()
}

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
    }
    
    sudoku_container.style.gridTemplateColumns = "repeat(" + size + ", " + 10 + "px)";
    sudoku_container.style.gridTemplateRows = "repeat(" + size + ", " + 10 + "px)";
    
        
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            var cell_to_insert = document.createElement('div');
            cell_to_insert.setAttribute('id', 'cell ' + i + "-" + j)
    
            var set_row = parseInt(i/(Math.sqrt(size)))
            var set_column = parseInt(j / Math.sqrt(size))
            
            if (size == 12) {
                set_row = parseInt(i/(Math.sqrt(size) + 1))
            }
            
    
            if (board_values[i][j] == 0)
                cell_to_insert.innerHTML = '';  
            else 
                cell_to_insert.innerHTML = board_values[i][j]
            
            
            if ((set_row + set_column) % 2 == 1) {
                cell_to_insert.style.backgroundColor = "#D3D3D3"
            }
            sudoku_container.appendChild(cell_to_insert);
        }
    }
}

main()

