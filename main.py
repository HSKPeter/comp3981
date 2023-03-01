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

@app.get("/board")
def load_board():
    return {"board": puzzle_loader.load(9)}

@app.get("/brute-force")
def solve_brute_force():
    board = [[]]
    return {"board": board}

@app.get("/csp")
def csp():
    board = [[]]
    return {"board": board}