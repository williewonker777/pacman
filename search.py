#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# search.py
# ---------
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



"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    # util.py에서 제공하는 Stack 사용
    stack = util.Stack()
    visited = set()  # 이미 방문한 상태 저장. 상태가 hashable하다는 가정 하에 사용
    # 초기 상태와 빈 액션 목록을 스택에 푸시
    stack.push((problem.getStartState(), []))

    while not stack.isEmpty():
        state, actions = stack.pop()
        if problem.isGoalState(state):
            return actions
        if state not in visited:
            visited.add(state)
            for next_state, action, cost in problem.getSuccessors(state):
                if next_state not in visited:
                    # 현재까지의 액션 목록에 새 액션을 추가하여 경로 구성
                    stack.push((next_state, actions + [action]))
    return []  # 목표 상태를 찾지 못한 경우 빈 리스트 반환
    

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # util.py에서 제공하는 Queue 사용
    queue = util.Queue()
    visited = set()
    start = problem.getStartState()
    queue.push((start, []))
    visited.add(start)

    while not queue.isEmpty():
        state, actions = queue.pop()
        if problem.isGoalState(state):
            return actions
        for next_state, action, cost in problem.getSuccessors(state):
            if next_state not in visited:
                visited.add(next_state)
                queue.push((next_state, actions + [action]))
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # util.py의 PriorityQueue 사용 (우선순위: 누적 비용)
    pq = util.PriorityQueue()
    start = problem.getStartState()
    # 초기 상태: (상태, 액션 리스트, 누적 비용), 우선순위 = 누적 비용
    pq.push((start, [], 0), 0)
    # visited 딕셔너리: 각 상태에 대해 현재까지의 최저 누적 비용 기록
    visited = {start: 0}

    while not pq.isEmpty():
        state, actions, costSoFar = pq.pop()
        if problem.isGoalState(state):
            return actions
        for next_state, action, step_cost in problem.getSuccessors(state):
            new_cost = costSoFar + step_cost
            # 이미 방문한 상태가 없거나 더 낮은 비용으로 도달한 경우 갱신
            if next_state not in visited or new_cost < visited[next_state]:
                visited[next_state] = new_cost
                pq.push((next_state, actions + [action], new_cost), new_cost)
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    """
    A*: 누적 비용과 휴리스틱 값을 합한 값이 가장 낮은 노드를 우선적으로 탐색합니다.
    """
    pq = util.PriorityQueue()
    start = problem.getStartState()
    initial_priority = heuristic(start, problem)
    pq.push((start, [], 0), initial_priority)
    visited = {start: 0}

    while not pq.isEmpty():
        state, actions, costSoFar = pq.pop()
        if problem.isGoalState(state):
            return actions
        for next_state, action, step_cost in problem.getSuccessors(state):
            new_cost = costSoFar + step_cost
            if next_state not in visited or new_cost < visited[next_state]:
                visited[next_state] = new_cost
                priority = new_cost + heuristic(next_state, problem)
                pq.push((next_state, actions + [action], new_cost), priority)
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
