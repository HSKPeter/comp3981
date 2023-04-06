# COMP 3981 - Introduction to Artificial Intelligence and Machine Learning
This is the repository for COMP 3981 (Introduction to Artificial Intelligence and Machine Learning) project. This project is the result of work by [Patrick Cammayo](https://www.linkedin.com/in/patrick-cammayo-8a535026a/), [Peter Hui](https://www.linkedin.com/in/peter-h-84316221b/), [Sepehr Zohoori Rad](https://sepzie.github.io/), and [Simar Vashisht](https://www.linkedin.com/in/simar-vashisht/).
This project is a web application that solves Sudoku puzzles. The web application is built with HTML, CSS, JavaScript and [FastAPI](https://fastapi.tiangolo.com/).

For the Sudoku solver, we have implemented two algorithms: brute force and constraint satisfaction problem (CSP).  The brute force algorithm is implemented with the depth first search (DFS) approach, and the CSP algorithm is implemented with the backtracking search approach.
Both algorithms and configured to solve puzzles of size 9x9, 12x12, 16x16, and 25x25.  The CSP algorithm is also configured to solve puzzles of size 100x100.

## Getting started 🚀

1. Clone this git repository
   ``` sh
   git clone https://github.com/HSKPeter/comp3981.git
   ```

2. [Optional 👀] Create a Python virtual environment
   - In the project repo, create a virtual environment by running `python -m venv ./venv` or `python3 -m venv ./venv`
   - Run the virtual environment
     - Windows: Run `.\venv\Scripts\activate.bat`
     - Mac: Run `source venv/bin/activate`

3. Install dependencies by running `pip install -r requirements.txt`

4. Run the web backend server by running one of the following commands in terminal:
   - `uvicorn main:app --reload` 
   - `python -m uvicorn main:app --reload`
   - `python3 -m uvicorn main:app --reload`

5. Host the frontend with the [live server extension](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer) in VS code

6. Visit http://127.0.0.1:5500/ui/html/main_menu.html to go to the UI of the sudoku solver.

## Brute Force Algorithm

### Depth First Search (DFS)
We have implemented the brute force algorithm with the depth first search (DFS) approach, by maintaining a stack to store nodes that are yet to be explored.

In each iteration, we peak the top node from the stack, and check if the node is a goal node.  If it is, we return the node as the solution.  Otherwise, we expand the node by generating its children nodes.  

### MRV
During node expansion, we would select the cell with the least number of remaining possible values.  After that, for each possible value of that selected empty cell, we would generate a child node that represents the board state after inserting that value into the selected empty cell.

### Minimum assigned neighbours as tie-breaking rule
In the case where there are multiple empty cells with the same number of remaining possible values, we would select the cell with the least number of assigned neighbours.  "Neighbour" here refers to the cells that are in the same row, column, or sub-square as the selected cell.  After some experiments, we found that this tie-breaking rule could help to solve the board with less number of guesses.

### Prioritize children nodes with smaller total domain size
For each children node, we would calculate the total domain size of the board, which is the sum of the number of remaining possible values of all the empty cells.  We would then sort the children nodes based on the total domain size, and push the child node with the smallest total domain size into the stack.

### Backtrack mechanism
In this depth first search implementation, backtrack would be taken place when we have explored all the children nodes of the node that is at the top of the stack.  We would then pop that node from the stack, and continue the iteration by exploring the next top node in the stack, which is the parent node of the node that we have just popped.

### Reserved stack
One of the challenges of the brute force algorithm is that it might get stuck in a local minimum.  To address this issue, we have set a custom timeout counter.  When it reaches a certain time limit, the algorithm would stop the current search, and migrate all the current nodes into a reserved stack.  Then, the algorithm would continue the search, starting from one of the child node of the root node.  If all the children nodes of the root node has been visited, then all nodes in the reserved stack would be migrated back to the main stack, the search would go on until a solution is found or all nodes has been evaluated. 

## CSP
The brute force algorithm has its limitation in solving large and difficult boards, and this brings us to the CSP algorithm, which is more powerful to solve sudoku problems, and also more promising to solve difficult 25x25 boards.

### Degree and MRV
In the CSP algorithm, we use a combination of the Minimum Remaining Values (MRV) and Degree heuristics to select unassigned variables for assignment.

MRV: The algorithm first applies the MRV heuristic, which chooses the variable with the fewest legal values remaining in its domain. By selecting variables that have fewer possibilities, we can reduce the search space and minimize the chances of backtracking.

Degree: If there is a tie for MRV, the algorithm uses the Degree heuristic as a tiebreaker. The Degree heuristic selects the variable involved in the highest number of constraints with other unassigned variables. By prioritizing variables with higher constraint involvement, we can minimize the impact of current assignments on future ones, further improving the efficiency of the algorithm.

### Least Constraining Value (LCV)
To determine the order of the values on a variable for which value to attempt first, we use the least constraining value. The idea of this heuristic is to assign a value that imposes the least impact on it's neighbouring cells in order to minimize the impact of the current assignment to the future assignment to other variables. By choosing that value that eliminates the fewest options for other variales, this heuristic helps to avoid unnecessary backtracking and increases the efficiency of the algorithm.

### MAC heuristics based on AC-3
We have also used the Maintaining Arc Consistency (MAC) heuristic, which is based on the AC-3 algorithm, to infer the domains of the cells in the Sudoku puzzle.  As such, cells domains would be updated whenever a new value is assigned to a cell, and arc consistency could be always maintained.  This would help to prune the search tree, and make the CSP algorithm more efficient.

### Multiprocessing
To use all the processing power of the machine, we have implemented multiprocessing in the CSP algorithm. The multiprocessing is done by first expanding the root node, and then running the CSP algorithm on each of the children nodes in parallel. The algorithm would then return the first solution that is found, and terminate the other processes.

## Table Results (All Samples)
| **Size** | **Average Time** | **Standard Deviation** |
|----------|------------------|------------------------|
| 9x9 | 0.2s | 2.3s |
| 12x12 | 2.3s | 2.3s | 
| 16x16 | 2.3s | 2.3s |
| 25x25 | 63.2s | 42.8s |


## Challenge of solving 100x100 Sudoku
At the moment, we are unable to solve a 100x100 Sudoku with the CSP algorithm.  In this section, we will discuss the challenges that we have encountered, and the efforts that we have made.

We understand that solving a 100x100 sudoku could be computationally expensive, and thus we have attempted to run the CSP algorithm on Microsoft Azure, by taking the following steps:

1. Apply for [free student credits](https://azure.microsoft.com/en-us/free/students/) from Microsoft Azure

2. Setup a [Linux virtual machine (VM)](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/quick-create-portal), and store the corresponding `pem` file in a safe place

3. Connect to the VM instance using [SSH](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/ssh-from-windows#connect-to-your-vm)
``` txt
ssh -i <path of pem file> azureuser@<public IP address of VM instance>
```

4. Setup the environment in the VM instance with the following commands:
   ``` sh
   git clone https://github.com/HSKPeter/comp3981.git
   cd comp3981
   pip install -r requirements.txt
   ```

5. Run the CSP algorithm with the following command:
   ``` sh
   # setup a screen session to run the program, so that it would not be terminated when we disconnect from the VM instance
   screen

   # run the CSP algorithm, which is customized to run on Azure
   python3 csp_azure.py 
   ```

During the first few attempts of running the CSP algorithm, we kept encountering the **out of memory (OOM)** issue.  It is likely because each node in our CSP algorithm stores a a dictionary that represents the domain of each board cell, and the memory usage increases drastically as the board size increases up to 100x100.  As such, we further refactored the CSP algorithm, so that the nodes are stored as json files in file storage, and the algorithm only loads the nodes that are needed for the current iteration.  

However, we still encountered the same OOM issue.  We then decided to externalize the memory by using the Azure Storage to store the json files.  After that, the algorithm is able to undergo iterations without the OOM issue. However, it could take more than 6 minutes for a single iteration, and this is due to the MAC heuristics, which could become computationally expensive as the board size gets as large as 100x100.

As such, we believe that our CSP algorithm at the moment is not powerful to solve a 100x100 Sudoku within a reasonable amount of time.


## References
- [A study of Sudoku solving algorithms](https://www.csc.kth.se/utbildning/kth/kurser/DD143X/dkand12/Group6Alexander/report/PATRIK_BERGGREN_DAVID_NILSSON.rapport.pdf) 
- [Solving Sudoku by Heuristic Search](https://medium.com/@davidcarmel/solving-sudoku-by-heuristic-search-b0c2b2c5346e)
- [A Sudoku Solver - Mike Schermerhorn](https://www.cs.rochester.edu/u/brown/242/assts/termprojs/Sudoku09.pdf)
- [Study of Brute Force and Heuristic Approach to Solve Sudoku](https://www.ijettcs.org/Volume4Issue5(2)/IJETTCS-2015-10-10-17.pdf)
- [Sudoku Solver and Generator used for generating solved sudoku puzzles up to 25x25](https://github.com/dangnguyendota/SudokuGeneratorAndSolver)
- [Code snippet for generating 100x100 sudoku puzzles](https://stackoverflow.com/a/56581709)
