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
            const rows = textContent.trim().split('\n');
            var board = [];
            for (let i = 0; i < rows.length; i++) {
                const rowValues = rows[i].trim().split(',').map(value => parseInt(value));
                board.push(rowValues);
            }
            const size = getBoardSize(board);
            localStorage.setItem("board", JSON.stringify(board));
            localStorage.setItem("size", size);
            console.log("Board: ", localStorage.getItem("board"));
            console.log("Size: ", localStorage.getItem("size"))
            window.location.href = "solver_screen.html?source=file"
        };
    }
});

const isValidLength = (num) => {
    const acceptableValues = [9, 12, 16, 25, 100];
    return acceptableValues.includes(num);
}

const isValidTextContent = (textContent) => {
    try {
        console.log("Inside isValidTextContent!");

        

        

        return isValidLength(rowCount) && rowCount === columnCountOfFirstRow;

    } catch (error) {
        return false;
    }
}

const getBoardSize = (boardInput) => {
    return boardInput[0].length;
}