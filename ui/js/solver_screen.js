
import SudokuPuzzleGenerator from "./sudoku_generator.js";

const timeContainer = document.getElementById("timeContainer");
const timeText = document.getElementById("time")
const title = document.getElementById("title")
const spinner = document.getElementById("spinner")
const bruteForceButton = document.getElementById("solve-brute-force")
const cspButton = document.getElementById("solve-csp")
const clearButton = document.getElementById("clear")


const FAIL = "Fail"
const SUCCESS = "Success"

let deltaTime;
var startTime;
let endTime;
let intervalId;

bruteForceButton.addEventListener("click", function () {
    displayTime(solveWithBruteForce);
});
cspButton.addEventListener("click", function () {
    displayTime(solveWithCSP);
});
clearButton.addEventListener("click", function () {
    window.location.href = "main_menu.html";
});

async function solve(algorithm) {
    spinner.style.display = "block"
    disableButtons()
    localStorage.setItem("Algorithm", algorithm);
    const boardJsonString = localStorage.getItem("Board");
    const parsedBoard = JSON.parse(boardJsonString);

    const fetchConfig = {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ board: parsedBoard })
    };

    const apiRoute = formatAlgorithmNameToApiRoute(algorithm);
    const response = await fetch(`http://localhost:8000/${apiRoute}/`, fetchConfig);
    spinner.style.display = "none"
    if (response.status === 404) {
        const { message } = await response.json();
        alert(message);

        throw Error()
        
        // myReject()
        // return;
    }
    const { board, duration } = await response.json();
    // myResolve()

    fillBoard(board)
    stopDynamicTimer({duration, status: SUCCESS})
}

function formatAlgorithmNameToApiRoute(algorithmName) {
    return algorithmName?.toLowerCase()?.split(" ")?.join("-");
}

async function solveWithBruteForce() {
    await solve("Brute Force")
}


async function solveWithCSP() {
    await solve("CSP")
}

function displayTime(solveAlgorithm) {
    startTime = Date.now()
    intervalId = setInterval(updateTime, 10);
    solveAlgorithm()
    .then(enableButtons)
    .catch(() => {
        stopDynamicTimer({status: FAIL})
        enableButtons()
    })
    // endTime = Date.now();
    // deltaTime = endTime - startTime;
    // timeText.innerText = deltaTime;
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

function stopDynamicTimer({duration, status}) {
    clearInterval(intervalId);
    if (status === FAIL) {
        document.getElementById("time-title").style.color = "#FF0000"
    } else {
        document.getElementById("time-title").style.color = "#228C22"
    }
    
    intervalId = null;

    if (duration) {
        showTime(duration)
    }
}

function updateTime() {
    const elapsedTime = Date.now() - startTime;
    const minutes = Math.floor(elapsedTime / (1000 * 60));
    const seconds = Math.floor((elapsedTime % (1000 * 60)) / 1000);
    const milliseconds = Math.floor((elapsedTime % 1000) / 10);
    const formattedTime = `${padNumber(minutes)}:${padNumber(seconds)}:${padNumber(milliseconds)}`
    showTime(formattedTime)
}

function showTime(timeToDisplay) {
    document.getElementById("time").textContent = timeToDisplay;
}

function padNumber(num) {
    return num.toString().padStart(2, "0");
}

function disableButtons() {
    cspButton.disabled = true
    bruteForceButton.disabled = true
    clearButton.disabled = true
}

function enableButtons() {
    cspButton.disabled = false
    bruteForceButton.disabled = false
    clearButton.disabled = false
}

main()

