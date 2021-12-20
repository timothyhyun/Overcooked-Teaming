
# Timothy Hyun
# November 19, 2021
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def calcProx (obj, player):
  objRange = [[obj[0]+1, obj[1]], [obj[0]-1, obj[1]], [obj[0], obj[1]+1], [obj[0], obj[1]-1]]
  if player in objRange:
    return False
  else:
    return True

def findNewObj (state, player, next):
  objs = state["objects"]
  objsNext = next["objects"]
  for i in list(objs.keys()):
    if (i not in list(objsNext.keys())):
      if calcProx(objs[i]["position"],state["players"][player]["position"]):
        return "NOT"
      if (objs[i]["name"] == "soup"):
        return "NOT"
      else:
        return i
  for i in list(objsNext.keys()):
    if (i not in list(objs.keys())):
      if calcProx(objsNext[i]["position"],state["players"][player]["position"]):
        return "NOT"
      if (objsNext[i]["name"] == "soup"):
        return "NOT"
      else:
        return i
  return "NOT"

def setActions (game):
  states = list(game["state"])
  nextStates = list(game['next_state'])
  times = list(game["time_elapsed"])
  actions = []
  for i in range(len(states)):
    tempState = states[i]
    tempNext = nextStates[i]
    player0 = tempState["players"][0]
    player1 = tempState["players"][1]
    try: 
      player0held = player0["held_object"]
    except:
      player0held = []
    try: 
      player1held = player1["held_object"]
    except:
      player1held = []
  
    player0next = tempNext["players"][0]
    player1next = tempNext["players"][1]
    try: 
      player0nextheld = player0next["held_object"]
    except:
      player0nextheld = []
    try: 
      player1nextheld = player1next["held_object"]
    except:
      player1nextheld = []
    if ((player0held == [] and player0nextheld != []) or (player0held != [] and player0nextheld == [])):
      if player0held != []:
        name = player0held["name"]
        newPos = findNewObj(tempState, 0, tempNext)
        if (newPos != "NOT"):
          actions.append((0,name,newPos,True, times[i]))
      else:
        name = player0nextheld["name"]
        newPos = findNewObj(tempState, 0, tempNext)
        if (newPos != "NOT"):
          actions.append((0,name,newPos,False, times[i]))
    if ((player1held == [] and player1nextheld != []) or (player1held != [] and player1nextheld == [])):
      if player1held != []:
        name = player1held["name"]
        newPos = findNewObj(tempState, 1, tempNext)
        if (newPos != "NOT"):
          actions.append((1,name,newPos,True, times[i]))
      else:
        name = player1nextheld["name"]
        newPos = findNewObj(tempState, 1, tempNext)
        if (newPos != "NOT"):
          actions.append((1,name,newPos,False, times[i]))
  return actions

def calcDelay (actions): 
# Iterate through actions list
# true = place
# false = pick up
  delays = []
  for action in actions:
    (player,name,ID,ac,time) = action
    if ac:
      for a in actions:
        (tempP, tempN, tempID, tempAC, tempTime) = a
        if ID == tempID and (not tempAC) and tempTime > time and name == tempN:
          if player != tempP:
            delays.append(tempTime - time)
            actions.remove(a)
            actions.remove(action)
            print(a)
            print(action)
            break
          else:
            actions.remove(a)
            actions.remove(action)
            break
  return delays


def calcHandoff(df):

  df["time_diff"] = df['time_elapsed'].diff()
  df = df[df["time_diff"]>0]
  df = df[df["time_diff"]<1]
  type(list(df["state"])[0]["objects"].keys())
  df.to_csv("filterai.csv")


  rounds = df.groupby("round_num")
  totals = []
  scores = []
  for i in range(5):
    roundTotal = []
    roundScore = []
    t = rounds.get_group(i)
    temp = t.groupby("workerid_num")
    hold = df[df["round_num"] == i]["workerid_num"].unique()
    for j in hold:
      res = temp.get_group(j)
      resac = setActions(res)
      delays = calcDelay(resac)
      
        
      roundTotal.append(sum(delays))
      roundScore.append(max(res["score"]))
    totals.append(roundTotal)
    scores.append(roundScore)

  t0 = rounds.get_group(3)
  workers = t0.groupby("workerid_num")
  w0 = workers.get_group(46)
  res = setActions(w0)

  calcDelay(res)

  roundNum = 0
  plt.scatter(scores[roundNum], totals[roundNum])
  plt.xlabel("Score")
  plt.ylabel("Functional Delay")
  plt.title("Functional Delay: Handoff Time")

  plt.savefig('round0.png')
  stats.pearsonr(scores[roundNum], totals[roundNum])
