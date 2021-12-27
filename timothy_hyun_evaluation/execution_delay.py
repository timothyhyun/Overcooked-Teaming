
# Timothy Hyun
# December 19, 2021
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import json


# Scenario 1: Trying to prep but cannot, pots are both full and player picks up an onion. 
# 1a: Waiting for partner to empty a single pot
# 1b: Waiting til own player drops the onion


# need to use ai-human trialsa


def startCond (state, nextState):
  objs = state["objects"]
  objsNext = nextState["objects"]
  numPots = 0
  nextPots = 0
  for ob in list(objs.keys()):
    temp = objs[ob]
    try:
      if temp["state"][0] == "onion" and temp["state"][1] == 3 and temp["state"][2] >= 20:
        numPots += 1
    except: 
      pass
  for ob in list(objsNext.keys()):
    temp = objsNext[ob]
    try:
      if temp["state"][0] == "onion" and temp["state"][1] == 3 and temp["state"][2] >= 20:
        nextPots += 1
    except: 
      pass
  players = state["players"]
  nextPlayers = nextState["players"]
  try:
    if (players[0]["held_object"]["name"] == "onion"):
      player0held = True
    else:
      player0held = False
  except: 
    player0held = False
  try:
    if (players[1]["held_object"]["name"] == "onion"):
      player1held = True
    else:
      player1held = False
  except:
    player1held = False
  try:
    if (nextPlayers[0]["held_object"]["name"] == "onion"):
      nextplayer0held = True
    else:
      nextplayer0held = False
  except: 
    nextplayer0held = False
  try:
    if (nextPlayers[1]["held_object"]["name"] == "onion"):
      nextplayer1held = True
    else:
      nextplayer1held = False
  except:
    nextplayer1held = False
  multiplier = 0
  # condition 1: pots are full and player picks up onion
  if (nextPots == 2 and nextplayer0held and not player0held):
    multiplier += 1
  if (nextPots == 2 and nextplayer1held and not player1held):
    multiplier += 1
  if (multiplier > 0):
    return multiplier
  # condition 2: player has onion and pots become done
  if (player0held and nextplayer0held and nextPots == 2 and numPots <2):
    multiplier += 1
  if (player1held and nextplayer1held and nextPots == 2 and numPots <2):
    multiplier += 1
  return multiplier

def endCond (state, nextState):
  objs = state["objects"]
  objsNext = nextState["objects"]
  players = state["players"]
  nextPlayers = nextState["players"]
  numPots = 0
  nextPots = 0
  for ob in list(objs.keys()):
    temp = objs[ob]
    try:
      if temp["state"][0] == "onion" and temp["state"][1] == 3 and temp["state"][2] >= 20:
        numPots += 1
    except: 
      pass
  for ob in list(objsNext.keys()):
    temp = objsNext[ob]
    try:
      if temp["state"][0] == "onion" and temp["state"][1] == 3 and temp["state"][2] >= 20:
        nextPots += 1
    except: 
      pass
  try:
    if (players[0]["held_object"]["name"] == "onion"):
      player0held = True
    else:
      player0held = False
  except: 
    player0held = False
  try:
    if (players[1]["held_object"]["name"] == "onion"):
      player1held = True
    else:
      player1held = False
  except:
    player1held = False
  try:
    if (nextPlayers[0]["held_object"]["name"] == "onion"):
      nextplayer0held = True
    else:
      nextplayer0held = False
  except: 
    nextplayer0held = False
  try:
    if (nextPlayers[1]["held_object"]["name"] == "onion"):
      nextplayer1held = True
    else:
      nextplayer1held = False
  except:
    nextplayer1held = False
  # condition 1: player drops onion to get pot
  if (nextPots == 2 and not nextplayer0held and player0held):
    return True
  if (nextPots == 2 and not nextplayer1held and player1held):
    return True
  # condition 2: pot gets emptied
  if (nextPots == 2 and numPots <2):
    return True
  return False

#instead iteration through because delay cannot happen concurrently
def setActions (game):
  ifStart = False
  states = list(game["state"])
  nextStates = list(game["next_state"])
  times = list(game["time_elapsed"])
  recordedTimes = []
  numDelays = 0
  res = 0
  for i in range(len(states)):
    tempState = states[i]
    tempNext = nextStates[i]
    tempDelays = startCond(tempState, tempNext)
    for j in range(tempDelays):
      recordedTimes.append(times[i])
    numDelays += tempDelays
    if (numDelays > 2):
      print("Something is wrong")
    if (numDelays > 0 and endCond(tempState, tempNext)):
      numDelays = 0
      for time in recordedTimes:
        res += (times[i]-time)
      recordedTimes = []
  return res



def executionDelay (df):
  df["time_diff"] = df['time_elapsed'].diff()
  df = df[df["time_diff"]>0]
  df = df[df["time_diff"]<1]
  rounds = df.groupby("layout_name")
  totals = []
  scores = []
  atotals = []
  ascores = []
  for i in df["layout_name"].unique():
    roundTotal = []
    roundScore = []
    t = rounds.get_group(i)
    temp = t.groupby("workerid_num")
    hold = df[df["layout_name"] == i]["workerid_num"].unique()
    for j in hold:
      res = temp.get_group(j)
      roundTotal.append(setActions(res))
      atotals.append(setActions(res))
      ascores.append(max(res["score"]))
      roundScore.append(max(res["score"]))
    totals.append(roundTotal)
    scores.append(roundScore)
  return totals

def plotExecution(scores, totals, roundNum):
  plt.scatter(scores[roundNum], totals[roundNum])
  plt.xlabel("Score")
  plt.ylabel("Functional Delay")
  plt.title("Functional Delay: Execution Delay")

  plt.savefig('round0.png')
  stats.pearsonr(scores[roundNum], totals[roundNum])
