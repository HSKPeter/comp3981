'use strict';

var generate_button = document.getElementById("generate_button");
var cancel_button = document.getElementById("cancel_button");
var select_element = document.getElementById("board_size");

// Sends the user back to the main menu if the 'cancel' button is clicked.
cancel_button.addEventListener('click', function() {
    window.location.href = "main_menu.html"
})

/**
 * Creates an onClickListener that takes the board size selected and goes to a screen with that board size.
 * If the board size is greater than 25x25 then the user is taken to a screen without the Brute Force button.
 */
generate_button.addEventListener('click', function() {
    var selected_board_size = parseInt(select_element.value);
    localStorage.setItem("size", selected_board_size)
    if (selected_board_size > 25) {
        window.location.href = "solver_screen_no_brute.html"
    } 
    else {
        window.location.href = "solver_screen.html";
    }    
})


    