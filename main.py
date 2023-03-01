from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from puzzle_loader import *

app = FastAPI()
puzzle_loader = PuzzleLoader()

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
    board =  puzzle_loader.load(size)
    return {"board": board}

@app.get("/brute-force")
def solve_brute_force():
    board = [[]]
    return {"board": board}

@app.get("/csp")
def csp():
    board = [[]]
    return {"board": board}