# COMP 3981

## Getting started 🚀

1. Clone this git repository

``` sh
git clone https://github.com/HSKPeter/comp3981.git
```

2. [Optional 👀] Create a Python virtual environment
   - In the project repo, create a virtual environment by running `python3 -m venv ./venv` in your terminal
   - Run the virtual environment
     - Windows: Run `.\venv\Scripts\activate.bat`
     - Mac: Run `source venv/bin/activate`

3. Install dependencies by running `pip install -r requirements.txt`

4. Run the web backend server by running `uvicorn main:app --reload` in terminal or 'python -m uvicorn main:app --reload'

5. Host the frontend with the live server extension in VS code

## Brute Force Algorithm

### Design of heuristics
- TBC

### Limitation
- TBC

## CSP

### Degree and MRV
- TBC

### MAC heuristics based on AC-3
- TBC

## Configuring AWS EC2
1. `chmod 400 <your pem filename>`
2. `ssh -i <your pem filename> ec2-user@<public IP address of EC2 instance>`
3. `sudo yum update`
4. `sudo yum install git`
5. `python3 --version`
6. `git clone https://github.com/HSKPeter/comp3981.git`
7. `cd comp3981`
8. 

## References
- [A study of Sudoku solving algorithms](https://www.csc.kth.se/utbildning/kth/kurser/DD143X/dkand12/Group6Alexander/report/PATRIK_BERGGREN_DAVID_NILSSON.rapport.pdf) 
- [Solving Sudoku by Heuristic Search](https://medium.com/@davidcarmel/solving-sudoku-by-heuristic-search-b0c2b2c5346e)
- [A Sudoku Solver - Mike Schermerhorn](https://www.cs.rochester.edu/u/brown/242/assts/termprojs/Sudoku09.pdf)
- [Study of Brute Force and Heuristic Approach to
Solve Sudoku](https://www.ijettcs.org/Volume4Issue5(2)/IJETTCS-2015-10-10-17.pdf)





