from fastapi import FastAPI
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from puzzle_loader import *
from sudoku_solver_brute_force import solve_with_brute_force, SolverExecutionExpiredException, PuzzleUnsolvedException
from pydantic import BaseModel
from typing import List
from sudoku_solver_csp_recursive import solve_with_csp_recursive
from sudoku_solver_csp_iterative import solve_with_csp_iterative
import time
from uuid import uuid4
import concurrent.futures
import asyncio

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

@app.middleware("http")
async def extend_sse_timeout(request, call_next):
    response = await call_next(request)
    if isinstance(response, Response) and "text/event-stream" in response.headers.get("content-type", ""):
        response.headers["cache-control"] = "no-cache"
        response.headers["connection"] = "keep-alive"
        response.headers["keep-alive"] = "timeout=300"
    return response

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/board/{size}")
def load_board(size: int):
    global board
    
    board =  puzzle_loader.load_from_2d_array_txt_file(size)
    maskedBoard = puzzle_loader.mask_puzzle(board)
    return {"board": maskedBoard}

def convert_seconds_to_formatted_time(seconds):
    min = int(seconds // 60)
    sec = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{min:02}:{sec:02}.{ms:05}"

@app.post("/brute-force")
async def solve_brute_force(board_puzzle: BoardPuzzleData):
    board = board_puzzle.board
    ref_id = int(time.time()) + uuid4().hex
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(save_brute_force_solution, board, ref_id)
            
    return {"ref_id": ref_id}

def save_brute_force_solution(board, ref_id):
    brute_force_solution = find_brute_force_solution(board, ref_id)
    with open("tmp/" + ref_id + ".json", "w") as file:
        file.write(json.dumps(brute_force_solution))


def find_brute_force_solution(board, ref_id):
    start_time = time.perf_counter()
    try:
        result = solve_with_brute_force(board)
        end_time = time.perf_counter()
        duration =  end_time - start_time
        return {"board": result, "duration": convert_seconds_to_formatted_time(duration), "status": "success"}
    except Exception:
        end_time = time.perf_counter()
        duration =  end_time - start_time
        return {"duration": convert_seconds_to_formatted_time(duration), "status": "failed"}
        

def error_response_with_message(message):
    return JSONResponse(content={"message": message}, status_code=404)


@app.post("/csp")
async def solve_csp(board_puzzle: BoardPuzzleData):
    try:
        start_time = time.perf_counter()
        board = board_puzzle.board
        result = solve_with_csp_iterative(board)
        end_time = time.perf_counter()
        duration =  end_time - start_time
        return {"board": result, "duration": convert_seconds_to_formatted_time(duration)}
    except Exception as e:
        print("error")
        print(e)
        return error_response_with_message("Solution not found within reasonable amount of time.")


@app.get("/solution/{ref_id}")
def get_solution(ref_id: str):
    # Generate the Server-Sent Events
    event_generator = generate_events(ref_id)

    # Return the Server-Sent Events as a StreamingResponse
    return StreamingResponse(event_generator, media_type='text/event-stream')
    

async def generate_events(ref_id: str):
    filename = "tmp/" + ref_id + ".json"

    while True:
        if os.path.exists(filename):
            with open(filename) as file:
                yield f"data: {json.load(file)}\n\n"
                return
        else:
            data = {'status': 'loading'}
            yield f"data: {json.dumps(data)}\n\n"
            
            await asyncio.sleep(1)