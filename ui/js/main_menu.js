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
    window.location.href = "solver_screen.html?source=generate";
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
                window.location.href = "solver_screen.html?source=file";
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

const isValidTextContent = (textContent) => {
    try {
        const board = JSON.parse(textContent);
        const columnCountOfFirstRow = board[0].length;

        if (!isValidLength(columnCountOfFirstRow)) {
            return false;
        }

        let rowCount = 0;

        for (const row of board) {
            let columnCounter = 0;

            for (const cell of row) {
                if (isNaN(cell)) {
                    return false
                }
                columnCounter ++;
            }

            if (columnCounter !== columnCountOfFirstRow) {
                return false;
            }

            rowCount ++;
        }

        return isValidLength(rowCount) && rowCount === columnCountOfFirstRow;

    } catch (error) {
        return false;
    }
}

const parseBoard = (boardInput) => {
    const board = JSON.parse(boardInput);
    const size = board[0].length;

    return {
        board: boardInput,
        size
    };    
}