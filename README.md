# PacmanAI
This is my implementation of a Reinforcement-Learning Agent for Pacman

Files that I edited:
- **valueIterationAgents.py**
- **qLearningAgents.py**
- **analysis.py**


How to run:
Run in the terminal, make sure you are currently in the correct directory

## Autograder (Receiving full points): 
`python autograder.py`

## To run a specific question

### Question 1 (Value Iteration Agent)
`python autograder.py -q q2`

### Question 2 (Bridge Crossing Analysis)
`python autograder.py -q q3`

### Question 3 (Policies)
`python autograder.py -q q3`

### Question 4 (Asynchronous Value Iteration)
`python autograder.py -q q4`

### NO QUESTION 5

### Question 6 (Q-Learning Agent)
`python autograder.py -q q6`

### Question 7 (Epsilon Greedy)
`python autograder.py -q q7`

### Question 8 (Revisited Bridge Crossing)
`python autograder.py -q q8`

### Question 9 (Q-Learning Agent with Pacman)
`python autograder.py -q q9`

to run our Q-Learning Agent on pacman, we use

`python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid`

-p: Agent used in pacman
-x: # of training games that will run
-n: total number of games that will run
-l: level layout (can use any layout in ~/layouts/)

To watch 10 training games, you can use

`python pacman.py -p PacmanQAgent -n 10 -l smallGrid -a numTraining=10`

### Question 10 (Approximate Q-Learning with Pacman)
`python autograder.py -q q10`

to run our Approximate Q-Agent on pacman, we use

`python pacman.py -p ApproximateQAgent -x 2000 -n 2010 -l smallGrid`

-p: Agent used in pacman
-x: # of training games that will run
-n: total number of games that will run
-l: level layout (can use any layout in ~/layouts/)


To test our feature extractor, you can use

`python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid`

Warning: this may take a few minutes to train on the larger layouts)


