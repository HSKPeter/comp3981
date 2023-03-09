
import SudokuPuzzleGenerator from "./sudoku_generator.js";

const timeContainer = document.getElementById("timeContainer");
const timeText = document.getElementById("time")
const title = document.getElementById("title")
const spinner = document.getElementById("spinner")

let deltaTime;
var startTime;
let endTime;
let intervalId;

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
async function solveWithBruteForce() {
    spinner.style.display = "block"
    localStorage.setItem("Algorithm", "Brute Force");
    const boardJsonString = localStorage.getItem("Board");
    const parsedBoard = JSON.parse(boardJsonString);

    const fetchConfig = {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ board: parsedBoard })
    };

    const response = await fetch("http://localhost:8000/brute-force/", fetchConfig);
    spinner.style.display = "none"
    if (response.status === 404) {
        const { message } = await response.json();
        alert(message);
        return;
    }
    const { board } = await response.json();

    fillBoard(board)
    stopDynamicTimer()
}


// Replace contents of this function with the real algorithm
async function solveWithCSP() {
    localStorage.setItem("Algorithm", "CSP")
    const response = await fetch("http://localhost:8000/brute-force/");
    const { board } = await response.json();
    fillBoard(board)
    stopDynamicTimer()
}

function displayTime(solveAlgorithm) {
    startTime = Date.now()
    solveAlgorithm()
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


const urlParams = new URLSearchParams(window.location.search);
const source = urlParams.get('source');
const size = localStorage.getItem("size");
const sudoku_container = document.getElementById('sudoku_container');


function fillBoard(board_values) {
    sudoku_container.innerHTML = ""
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            var cell_to_insert = document.createElement('div');
            cell_to_insert.setAttribute('id', 'cell ' + i + "-" + j)

            var set_row = parseInt(i / (Math.sqrt(size)))
            var set_column = parseInt(j / Math.sqrt(size))

            if (size == 12) {
                set_row = parseInt(j / 4)
                set_column = parseInt(i / 3)
            }

            if (size == 9 || size == 12) {
                cell_to_insert.style.fontSize = "20px";
            }

            if (size == 16) {
                cell_to_insert.style.fontSize = "15px";
            }

            if (size == 25) {
                cell_to_insert.style.fontSize = "11px";
            }

            if (size == 100) {
                cell_to_insert.style.fontSize = "5px";
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
    localStorage.setItem("Board", JSON.stringify(board_values));
}


async function main() {
    let board_values;
    intervalId = setInterval(updateTime, 10);
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

    if (size == 9 || size == 12) {
        sudoku_container.style.gridTemplateColumns = "repeat(" + size + ", " + 30 + "px)";
        sudoku_container.style.gridTemplateRows = "repeat(" + size + ", " + 30 + "px)";
    }

    if (size == 16) {
        sudoku_container.style.gridTemplateColumns = "repeat(" + size + ", " + 25 + "px)";
        sudoku_container.style.gridTemplateRows = "repeat(" + size + ", " + 25 + "px)";
    }

    if (size == 25) {
        sudoku_container.style.gridTemplateColumns = "repeat(" + size + ", " + 18 + "px)";
        sudoku_container.style.gridTemplateRows = "repeat(" + size + ", " + 18 + "px)";
        document.body.style.overflow = "scroll";
        document.body.style.height = "700px"
    }

    if (size == 100) {
        sudoku_container.style.gridTemplateColumns = "repeat(" + size + ", " + 8 + "px)";
        sudoku_container.style.gridTemplateRows = "repeat(" + size + ", " + 8 + "px)";
        document.body.style.overflow = "scroll";
        document.body.style.height = "1100px"
    }

    fillBoard(board_values)
}




function stopDynamicTimer() {
    clearInterval(intervalId);
    document.getElementById("time-title").style.color = "#228C22"
    intervalId = null;
}

function updateTime() {
    const elapsedTime = Date.now() - startTime;
    const minutes = Math.floor(elapsedTime / (1000 * 60));
    const seconds = Math.floor((elapsedTime % (1000 * 60)) / 1000);
    const milliseconds = Math.floor((elapsedTime % 1000) / 10);
    document.getElementById("time").textContent =
        `${padNumber(minutes)}:${padNumber(seconds)}:${padNumber(milliseconds)}`;
}

function padNumber(num) {
    return num.toString().padStart(2, "0");
}


main()

