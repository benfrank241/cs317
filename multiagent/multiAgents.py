# multiAgents.py
# --------------
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
import game
import pacman
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        ghostPos = currentGameState.getGhostPositions()
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # print(successorGameState, newPos, newFood, newGhostStates, newScaredTimes)
        # print(ghostPos)

        "*** YOUR CODE HERE ***"

        if successorGameState.isWin():
            return 100000
        # go after the food unless the ghost is close then run away
        currScore = successorGameState.getScore()
        score = 0

        # if a ghost is within x manhattan distance, run away
        ghostDisToMan = []
        for i in ghostPos:
            ghostDisToMan.append(manhattanDistance(newPos, i))

        if min(ghostDisToMan) < 5:
            return -1000000

        # distance to food
        fList = newFood.asList()
        foodDisToMan = []
        # for loop gets the closest food and returns a higher score in relation to how close the food is
        for i in fList:
            foodDisToMan.append(manhattanDistance(newPos, i))
        # -0.1 so that the min found gives the least amount of negative points
        # a food that is 10 away will have a score of -10 where food 1 away will return -1 score
        score += currScore - 0.1 * min(foodDisToMan)

        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        #value function that determines whether to call max or min based on the agent's number
        def value(state, newState, depth, agent):
            agents = state.getNumAgents()
            #if agent is pacman, call max
            if agent + 1 == agents:
                return maxValue(newState, depth+1) #returns the next hypothetical state, increased depth of 1
            #else call min
            else:
                return minValue(newState, depth, agent+1) #returns the next hypothetical state, same depth, and next agent (ghost)


        def maxValue(state, depth): #agent will always be 0 so no need to pass it in
            #get pacman's moves and initialize v as -inf
            moves, v = state.getLegalActions(0), -9999999
            #if the moves or gameState are terminal, return the utility of the state          
            if gameState.isWin() or gameState.isLose() or len(moves) == 0 or depth == self.depth:
                return (self.evaluationFunction(state), None)            

            #(for each successor of state)for each move, generate a new state and call value to get the utility of the state                               
            for move in moves:
                #generate the successor state
                successorState = state.generateSuccessor(0, move)
                #get the min utility of the successor state
                successorValue, _ = minValue(successorState, depth, 1) #1 is always ghost

                #v = max(v, successorValue). But we also need to keep track of the action that led to the max value
                v = max(v, successorValue)
                if v == successorValue:
                    maxAction = move
            return (v, maxAction)


        def minValue(state, depth, agent):
            #get ghost's moves and initialize v as inf
            moves, v = state.getLegalActions(agent), 9999999
            #if the moves or gameState are terminal, return the utility of the state
            if gameState.isWin() or gameState.isLose() or len(moves) == 0 or depth == self.depth:
                return (self.evaluationFunction(state), None)
            
            #(for each successor of state)for each move, generate a new state and call value to get the utility of the state
            for move in moves:
                #generate the successor state
                successorState = state.generateSuccessor(agent, move)
                #get the value of the next agent (this will recursively call min or max when it's the last agent)
                successorValue, _ = value(state, successorState, depth, agent)

                #v = min(v, successorValue). But we also need to keep track of the action that led to the min value
                v = min(v, successorValue)
                if v == successorValue:
                    minAction = move
            return (v, minAction)

        #driver with the initial state and depth of 0, maxValue bc first agent is pacman, [1] bc we only want the action
        return maxValue(gameState, 0)[1]

                






class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        print("OUTDATED FOLDER, DOCS > UPDATEDCS317")
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
