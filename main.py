from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from puzzle_loader import *
from sudoku_solver import solve_with_brute_force, SolverExecutionExpiredException, PuzzleUnsolvedException
from pydantic import BaseModel
from typing import List

app = FastAPI()
puzzle_loader = PuzzleLoader()
board = None

class BoardPuzzleData(BaseModel):
    board: List[List[int]]

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/board/{size}")
def load_board(size: int):
    global board
    
    board =  puzzle_loader.load(size)
    maskedBoard = puzzle_loader.mask_puzzle(board)
    return {"board": maskedBoard}

@app.post("/brute-force")
async def solve_brute_force(board_puzzle: BoardPuzzleData):
    try:
        board = board_puzzle.board
        result = solve_with_brute_force(board)
        return {"board": result}
    except Exception:
        error_message = "Solution not found by brute force algorithm within reasonable amount of time.  You may consider to use CSP algorithm instead."
        return error_response_with_message(error_message)    
    # except SolverExecutionExpiredException:
    #     error_message = "Solution not found by brute force algorithm within reasonable amount of time.  You may consider to use CSP algorithm instead."
    #     return error_response_with_message(error_message)
    # except PuzzleUnsolvedException:
    #     return error_response_with_message("No solution found.")

def error_response_with_message(message):
    return JSONResponse(content={"message": message}, status_code=404)

@app.get("/csp")
def csp():
    return {"board": board}