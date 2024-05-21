# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # creating a counter of q values similar to our value iteration agent, but indexes with (state, action), rather than just state
        self.values = util.Counter()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.values[(state, action)] # returns the q value stored in our values list, or a 0 if we have never seen this state


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legal = self.getLegalActions(state)
        bestVal = float('-inf') # lowest possible float to get overwritten after first iteration of loop
        if not legal: return 0.0 # return value of 0 if there are no legal moves (as asked for above)
        for action in legal: # for every legal action, determine the best Q value
            qVal = self.getQValue(state, action)
            if qVal > bestVal: # if Q value > best known value (min int to start), Q val becomes new best val
                bestVal = qVal
        return bestVal
                

        
    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legal = self.getLegalActions(state)
        bestAction = None # if there are no legal actions, will return None, otherwise the best action is found and returned
        bestVal = float('-inf') # starts as lowest possible float, always updated by first legal action loop
        for action in legal: # for all legal actions, take Q value of state and action
            qVal = self.getQValue(state, action)
            if qVal == bestVal: # if our Q Value is the same as our best Q value, random probability between actions on which to choose
                bestAction = random.choice([bestAction, action])
            elif qVal > bestVal: # if our Q Value is better than our current best known Q value, update variables
                bestAction = action
                bestVal = qVal
        return bestAction

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        prob = util.flipCoin(self.epsilon) # our coin flip makes this epsilon-greedy
        if prob: # if we flipped true, choose random action we can legally make
            action = random.choice(legalActions)
        else:  # if we flipped false, compute our best action from Q values
            action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        currQVal = self.getQValue(state, action) # current Q value when given state and action
        QReward =  (reward + self.computeValueFromQValues(nextState) * self.discount) # determining reward
        updatedQVal = ((1 - self.alpha) * currQVal) + (self.alpha * QReward) # determining our new Q value, multiplied by the Q-learning rate (alpha)
        self.values[(state, action)] = updatedQVal # update our Q values counter
        return

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action) # getting list of features at (state, action)
        weights = self.weights # creating a copy of our weights counter
        QValue = 0.0 # setting our Q value to 0
        for feature in features: # compute Q value for all features
            featureValue = (features[feature] * weights[feature])
            QValue = QValue + featureValue # add featureValue * waitValue to our Q Value
        return QValue # return our Q value (sum of all feature values * reward values)
            

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action) # getting list of features at (state, action)
        discount = self.discount # creating a variable for the discount amount
        rewardVal = (reward + discount * self.computeValueFromQValues(nextState)) # calculate the reward
        difference = rewardVal - self.getQValue(state, action) # calculate difference in value
        for feature in features: # for all of our features, calculate and update the value of weights at the feature index
            self.weights[feature] = self.weights[feature] + (self.alpha * difference * features[feature])
        return

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print("Episodes: " + str(self.numTraining))
            print("Epsilon: Exploration Rate - " + str(self.epsilon))
            print("Alpha: Learning Rate - " + str(self.alpha))
            pass
