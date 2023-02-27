const exitButton = document.querySelector("#exit");
const boardSizeSelectMenu = document.querySelector("#selectBoardSize");
const fileInput = document.querySelector("#fileInput");
const generateButton = document.querySelector("#generate");

window.addEventListener('load', () => {
    // Ensure the file input element does not cache any value
    document.querySelector("#fileInput").value = ""
});

exitButton.addEventListener('click', () => {
    const confirmExit = confirm("Exit program?");
    if (confirmExit) {
        window.location.href = "https://www.bcit.ca/";
    }
});

boardSizeSelectMenu.addEventListener('change', (e) => {
    generateButton.disabled = false;
});

generateButton.addEventListener('click', (e) => {
    localStorage.setItem("size", boardSizeSelectMenu.value);
    window.location.href = "solver_screen.html";
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file.type !== "text/plain") {
        alert("File must be in txt format");
        e.target.value = null;
    } else {
        const reader = new FileReader();
        reader.readAsText(file);
        reader.onload = () => {
            const textContent = reader.result;
            if (isValidTextContent(textContent)) {
                const {board, size} = parseBoard(textContent);
                localStorage.setItem("board", board);
                localStorage.setItem("size", size);
                window.location.href = "solver_screen.html";
            } else {
                alert("Invalid content of text file");
                e.target.value = null;
            }
        };
    }
});

const isValidLength = (num) => {
    const acceptableValues = [9 , 12, 16, 25, 100];
    return acceptableValues.includes(num);
}

const isAllNum = (text) => {
    const characters = text.split("");
    for (const char of characters) {
        if (isNaN(parseInt(char))) {
            return false;
        }
    }

    return true;
}

const isValidTextContent = (textContent) => {
    const lines = textContent.split("\n");

    const boardHeight = lines.length;
    const boardWidth = lines[0].length;


    if (boardHeight !== boardWidth || !isValidLength(boardWidth)) {
        return false;
    }
    
    for (let i = 0; i < lines.length; i ++) {
        // Ensure texts are all numbers
        if (!isAllNum(lines[i])) {
            return false;
        }

        // Ensure the length of each line is consistent with the board width
        if (lines[i].length !== boardWidth) {
            return false;
        }
    }

    return true;
}

const parseBoard = (boardInput) => {
    const rows = boardInput.split("\n");
    const board = [];

    for (const row of rows) {
        board.push(row.split("").map(numStr => parseInt(numStr)));
    }

    return {
        board,
        size: rows.length
    };
}