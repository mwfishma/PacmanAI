# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for iteration in range(self.iterations): # for every iteration of self possible
            newValues = self.values.copy # create a copy of our values counter
            states = self.mdp.getStates() # a list of all states
            qCounter = util.Counter() # a Q val counter
            for state in states: # for all states mdp returned
             actions = self.mdp.getPossibleActions(state) # determine all possible actions
             bestVal = float('-inf') 
             for action in actions:
                 qVal = self.computeQValueFromValues(state, action) # for every possible action in current state, compute Q value
                 if qVal > bestVal: # if our Q value is greater than our best known value for the current state, set Q val as our new best val
                     bestVal = qVal 
                 qCounter[state] = bestVal # put best value into our counter at index for state
            self.values = qCounter # update our values counter
        


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        discount = self.discount # needed for math later
        values = self.values
        total = 0 
        transC_P = self.mdp.getTransitionStatesAndProbs(state, action) # list of pairs of children states and probability of visitting
        for child, prob in transC_P: # for every child state and its probability, calculate the reward and add to the total
            reward = self.mdp.getReward(state, action, child)
            val = reward + (discount * values[child])
            total = total + (prob * val) # calculation for Q value
        return total

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestAction = None # best known action, if there are no possible actions, return None
        bestVal = float('-inf') # lowest possible value for the best value
        actions = self.mdp.getPossibleActions(state) # list of all possible actions for the state given in parameters
        for action in actions: # for every possible action, compute Q value
            qVal = self.computeQValueFromValues(state, action)
            if qVal > bestVal: # if our Q Val is better than the best known value
                bestVal = qVal
                bestAction = action
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):                        # REARRANGE CODE, ADD MORECOMMENTS, FIX
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates() # list of all states
        for iteration in range(self.iterations): # for every iteration of self
            state = states[iteration % len(states)] # determine the state we are on based off iteration and # of states
            terminal = self.mdp.isTerminal(state) # boolean for whether or not our current state is terminal
            if terminal: continue # if we are in a final state, move onto next iteration (prevents looping states)
            # if we were not in a terminal state, perform calculations and update
            qValList = [] # list of all possible Q values
            actions = self.mdp.getPossibleActions(state)
            for action in actions: # for every action possible, compute the Q values
                qValList.append(self.computeQValueFromValues(state, action)) # add our value to the list
            self.values[state] = max(qValList) # set the highest score Q value at our values counter at index state

            


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # ???

