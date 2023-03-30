from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from puzzle_loader import *
from sudoku_solver_brute_force import solve_with_brute_force, SolverExecutionExpiredException, PuzzleUnsolvedException
from pydantic import BaseModel
from typing import List
from sudoku_solver_csp_recursive import solve_with_csp_recursive
import time

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

def convert_seconds_to_formatted_time(seconds):
    min = int(seconds // 60)
    sec = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{min:02}:{sec:02}.{ms:05}"

@app.post("/brute-force")
async def solve_brute_force(board_puzzle: BoardPuzzleData):
    try:
        start_time = time.perf_counter()
        board = board_puzzle.board
        result = solve_with_brute_force(board)
        end_time = time.perf_counter()
        duration =  end_time - start_time
        return {"board": result, "duration": convert_seconds_to_formatted_time(duration)}
    except Exception:
        return error_response_with_message("Solution not found by brute force algorithm within reasonable amount of time.  You may consider to use CSP algorithm instead.")    
    # except SolverExecutionExpiredException:
    #     error_message = "Solution not found by brute force algorithm within reasonable amount of time.  You may consider to use CSP algorithm instead."
    #     return error_response_with_message(error_message)
    # except PuzzleUnsolvedException:
    #     return error_response_with_message("No solution found.")

def error_response_with_message(message):
    return JSONResponse(content={"message": message}, status_code=404)


@app.post("/csp")
async def solve_csp(board_puzzle: BoardPuzzleData):
    try:
        start_time = time.perf_counter()
        board = board_puzzle.board
        result = solve_with_csp_recursive(board)
        end_time = time.perf_counter()
        duration =  end_time - start_time
        return {"board": result, "duration": convert_seconds_to_formatted_time(duration)}
    except Exception as e:
        # print(e)
        return error_response_with_message("Solution not found within reasonable amount of time.")
