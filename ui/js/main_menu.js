"use strict";

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
        return
    }
    const reader = new FileReader();
    reader.readAsText(file);
    reader.onload = () => {
        let inputData = reader.result;
        const [board, isValid] = isValidTextContent(inputData);
        if (!isValid) {
            e = null;
            console.log("The file is invalid!")
            return
        }
        console.log("The file is valid!");
        const size = getBoardSize(board);
        localStorage.setItem("board", JSON.stringify(board));
        localStorage.setItem("size", size);
        console.log("Board: ", localStorage.getItem("board"));
        console.log("Size: ", localStorage.getItem("size"))
        window.location.href = "solver_screen.html?source=file"
    }
});

const isValidLength = (num) => {
    const acceptableValues = [9, 12, 16, 25, 100];
    return acceptableValues.includes(num);
}

const validCharacters = (boardValues, length) => {
    var acceptableValues = [];
    for (let i = 0; i <= length; i++) {
        acceptableValues.push(i.toString());
    }
    console.log("Acceptable Values: ", acceptableValues)
    console.log("Board Values:", boardValues)
    for (let i = 0; i < boardValues.length; i++) {
        for (let j = 0; j < boardValues.length; j++) {
            if (!acceptableValues.includes(boardValues[i][j])) {
                alert("The board must only contain numbers 0-9, 0-12, 0-16, 0-25, or 0-100.");
                console.log("Board Value:", boardValues[i][j]);
                console.log("Board has invalid characters!");
                return false;
            }
        }
    }
    return true;
}

const isValidTextContent = (fileContent) => {
    try {
        let lines = fileContent.split("\n");
        let values = [];
        for (let i = 0; i < lines.length; i++) {
            let line = lines[i].replace("\r", "");
            let lineValues = line.split(",");
            lineValues = lineValues.filter(function (value) {
                return value.trim() !== "";
            });
            values.push(lineValues);
        }
        if (values[values.length-1].length === 0) {
            values.pop();
        }
        console.log("Values: ", values);
        for (let i = 0; i < values.length; i++) {
            if (values[i].length !== values.length) {
                alert("The board must be square (same number of rows and columns).");
                return [null, false];
            }
        }
        if (!isValidLength(values.length)) {
            alert("The board must be 9x9, 12x12, 16x16, 25x25, or 100x100.");
            return [null, false];
        }
        if (!validCharacters(values, values.length)) {
            return [null, false];
        }
        return [values, true]

    } catch (error) {
        console.log("Error: ", error);
        return false;
    }
}

const getBoardSize = (boardInput) => {
    return boardInput[0].length;
}