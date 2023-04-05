# COMP 3981

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

## Brute Force Algorithm

### Depth First Search (DFS)

### Design of heuristics
- (MRV)
- (Tie-breaking factor: find cell with less assigned neighbours)
- (During node expansion, check validity of new insertion)

### Limitation
- (Limited capability to solve difficult 25x25 boards)

## CSP

### Degree and MRV
- TBC

### Least Constraining Value (LCV)
- TBC

### MAC heuristics based on AC-3
- TBC

### Multiprocessing
- TBC


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
- [Study of Brute Force and Heuristic Approach to
Solve Sudoku](https://www.ijettcs.org/Volume4Issue5(2)/IJETTCS-2015-10-10-17.pdf)