import asyncio
import concurrent.futures
import time
from typing import List
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

from puzzle_loader import *
from sudoku_solver_brute_force import solve_with_brute_force
from sudoku_solver_csp_iterative import solve_with_csp_iterative

app = FastAPI()
puzzle_loader = PuzzleLoader()
board = None
solutions = {}


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

    board = puzzle_loader.load_from_2d_array_txt_file(size)
    masked_board = puzzle_loader.mask_puzzle(board)
    return {"board": masked_board}


def convert_seconds_to_formatted_time(seconds):
    minutes = int(seconds // 60)
    sec = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{minutes:02}:{sec:02}.{ms:05}"


@app.post("/brute-force")
def solve_brute_force(board_puzzle: BoardPuzzleData):
    board = board_puzzle.board
    unix_epoch = int(time.time())
    ref_id = str(unix_epoch) + uuid4().hex
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(save_brute_force_solution, board, ref_id)

    return {"ref_id": ref_id}


def save_brute_force_solution(board, ref_id):
    solutions[ref_id] = {"result": None, "duration": None, "status": None}
    result, duration, status, msg = find_brute_force_solution(board)
    solutions[ref_id] = {"result": result, "duration": duration, "status": status, "msg": msg}


def find_brute_force_solution(board):
    start_time = time.perf_counter()
    try:
        result = solve_with_brute_force(board)
        end_time = time.perf_counter()
        duration = end_time - start_time
        return result, convert_seconds_to_formatted_time(duration), "success", None
    except Exception:
        end_time = time.perf_counter()
        duration = end_time - start_time
        return None, convert_seconds_to_formatted_time(
            duration), "failed", "Solution not found within reasonable amount of time."


def error_response_with_message(message):
    return JSONResponse(content={"message": message}, status_code=404)


@app.post("/csp")
async def solve_csp(board_puzzle: BoardPuzzleData):
    board = board_puzzle.board
    unix_epoch = int(time.time())
    ref_id = str(unix_epoch) + uuid4().hex
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, save_csp_solution, board, ref_id)

    return {"ref_id": ref_id}


def save_csp_solution(board, ref_id):
    result, duration, status, msg = find_csp_solution(board)
    solutions[ref_id] = {"result": result, "duration": duration, "status": status, "msg": msg}


def find_csp_solution(board):
    start_time = time.perf_counter()
    try:
        result = solve_with_csp_iterative(board)
        end_time = time.perf_counter()
        duration = end_time - start_time
        return result, convert_seconds_to_formatted_time(duration), "success", None
    except Exception:
        end_time = time.perf_counter()
        duration = end_time - start_time
        return None, convert_seconds_to_formatted_time(
            duration), "failed", "Solution not found within reasonable amount of time."


@app.get("/solution/{ref_id}")
def get_solution(ref_id: str):
    event_generator = generate_events(ref_id)
    return StreamingResponse(event_generator, media_type='text/event-stream')


async def generate_events(ref_id: str):
    while True:
        if solutions.get(ref_id) is not None:
            yield f"data: {json.dumps(solutions[ref_id])}\n\n"
            return
        else:
            data = {'status': 'loading'}
            yield f"data: {json.dumps(data)}\n\n"

            await asyncio.sleep(1)
