FLOOR_SQUARE_ROOTS = {
    9: 3,
    12: 3,
    16: 4,
    25: 5,
    100: 10
}


def get_sub_square_index(n, row, col):
    sub_n = FLOOR_SQUARE_ROOTS[n]  # number of subsquares in each row (3 for 12x12)
    sub_m = n // sub_n  # number of sub-squares in each column (4 for 12x12)
    sub_row = row // sub_n
    sub_col = col // sub_m
    sub_square_index = sub_row * sub_n + sub_col
    return sub_square_index
