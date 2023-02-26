const exitButton = document.querySelector("#exit");
const boardSizeSelectMenu = document.querySelector("#selectBoardSize");
const fileInput = document.querySelector("#fileInput");
const generateButton = document.querySelector("#generate");


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
                localStorage.setItem("board", parseBoard(textContent));
                window.location.href = "solver_screen.html";
            } else {
                alert("Invalid content of text file");
                e.target.value = null;
            }
        };
    }
});

const isValidTextContent = (textContent) => {
    return true; // TODO
}

const parseBoard = (boardInput) => {
    return boardInput; // TODO
    // return {
    //     size: 9,
    //     content: [
    //         []
    //     ]
    // }
}