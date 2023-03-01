from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from puzzle_loader import *

app = FastAPI()
puzzle_loader = PuzzleLoader()
board = None

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

@app.get("/brute-force")
def solve_brute_force():
    return {"board": board}

@app.get("/csp")
def csp():
    return {"board": board}