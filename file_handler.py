def board_to_file(filename):
    board = [[4, 0, 0, 0, 0, 0, 8, 0, 5], [0, 3, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 7, 0, 0, 0, 0, 0],
             [0, 2, 0, 0, 0, 0, 0, 6, 0], [0, 0, 0, 0, 8, 0, 4, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 6, 0, 3, 0, 7, 0], [5, 0, 0, 2, 0, 0, 0, 0, 0], [1, 0, 4, 0, 0, 0, 0, 0, 0]]
    result = ""

    for row in board:
        row_str = ",".join(str(num) for num in row)
        result += row_str + "\n"
    with open(f"assets/sample_files_for_upload/{filename}", "w") as file:
        file.write(result)


board_to_file("")
