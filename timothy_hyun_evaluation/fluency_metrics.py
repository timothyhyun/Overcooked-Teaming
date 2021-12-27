
# Timothy Hyun
# November 12, 2021
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
plt.rcParams.update({
   'font.size': 12
})




# Fluency #1: Human Idle Time: 
def humanIdle (time, index, state, next):
  if state["players"][index] == next["players"][index]:
    return time
  else:
    return 0


# Fluency #2: Robot Idle Time: 

def robotIdle (time, index, state, next):
  if index == 0:
    newindex = 1
  else:
    newindex = 0
  if state["players"][newindex] == next["players"][newindex]:
    return time
  else:
    return 0
  

# Fluency #3: Concurrent Activity 
def bothAct (x):
  tempHlist = [x["human_idle"], x["temp_hurange"], x['temp_hdrange']]
  tempRlist = [x["robot_idle"], x["temp_rurange"], x['temp_rdrange']]
  if 0 in tempHlist and 0 in tempRlist:
    return x["time_diff"]
  else:
    return 0


# Fluency #4: Functional Delay
# only if onion on counter and pot not full
# or dish on counter (index (2,_))
# random0 = forced coordination

def idling(state):
  temp = list(state["objects"].keys())
  idleOnion = False
  for obj in temp:
    if obj[0] == "2":
      if state["objects"][obj]["name"] == "dish":
        return True
      if state["objects"][obj]["name"] == "onion":
        idleOnion = True
  for obj in temp:
    if obj == "3,0" and idleOnion:
      return True
    if obj == "4,1" and idleOnion:
      return True
  return False

def functionalDelay(state, time):
  if idling(state):
    return time
  else:
    return 0


def robotInteract (time, action, index):
  if index == 0:
    index = 1
  else:
    index = 0
  if action[index] == 'INTERACT':
    return 1
  else:
    return 0

def humanInteract (time, action, index):
  if action[index] == 'INTERACT':
    return 1
  else:
    return 0


def fluencyMeasures (df):
  df["time_diff"] = df['time_elapsed'].diff()
  df = df[df["time_diff"]>0]
  df = df[df["time_diff"]<3]

  df["human_idle"] = df.apply(lambda x: humanIdle(x['time_diff'], x['player_index'], x['state'], x['next_state']), axis=1)
  df["robot_idle"] = df.apply(lambda x: robotIdle(x['time_diff'], x['player_index'], x['state'], x["next_state"]), axis=1)
  df["temp_hurange"] = df['human_idle'].shift(1)
  df['temp_hdrange'] = df['human_idle'].shift(-1)
  df["temp_rurange"] = df['robot_idle'].shift(1)
  df['temp_rdrange'] = df['robot_idle'].shift(-1)

  df["concurrent_activity"] = df.apply(lambda x: bothAct(x), axis=1)
  df.drop(columns=['temp_hurange', 'temp_hdrange', 'temp_rurange', 'temp_rdrange'], axis=1, inplace=True)
  df["functional_delay"] = df.apply(lambda x: functionalDelay(x['state'],x['time_diff']), axis=1)

  df["ai_interact"] = df.apply(lambda x: robotInteract(x['time_diff'], x['joint_action'], x['player_index']), axis=1)
  df["human_interact"] = df.apply(lambda x: humanInteract(x['time_diff'], x['joint_action'], x['player_index']), axis=1)

  # Contribution Metrics:
  contributionScores = []
  rounds = df.groupby("round_num")
  for i in range(5):
    temp = rounds.get_group(i)
    contributionScores.append((temp["human_interact"].sum(), temp["ai_interact"].sum()))
    
  contributionScores

  # scatter plots:
  scores = []
  fluency1 = []
  fluency2 = []
  fluency3 = []
  
  times = df.groupby("layout_name")
  for i in df["layout_name"].unique():
    t = times.get_group(i)
    temp = t.groupby("workerid_num")
    hold = df[df["layout_name"] == i]["workerid_num"].unique()
    t1 = []
    t2 = []
    t3 = []
    for j in hold:
      workTemp = temp.get_group(j)
      scores.append(workTemp["score"].max())
      t1.append(sum(list(workTemp["human_idle"])))
      t2.append(sum(list(workTemp["robot_idle"])))
      t3.append(sum(list(workTemp["concurrent_activity"])))
    fluency1.append(t1)
    fluency2.append(t2)
    fluency3.append(t3)


  return fluency1,fluency2,fluency3




def plotFluency (fluency1, fluency2,fluency3,scores):
  plt.scatter(scores, fluency1)
  plt.xlabel("Score")
  plt.ylabel("Human Idle Time")
  plt.title("Human Idle Time")
  stats.pearsonr(scores, fluency1)
  plt.savefig('fluency1.png')
  stats.pearsonr(scores, fluency1)

  plt.scatter(scores, fluency2)
  plt.title("Robot Idle Time: Asymmetric Advantages", fontsize=20)
  plt.xlabel("Score")
  plt.ylabel("Robot Idle Time")

  plt.savefig('fluency2.png')
  stats.pearsonr(scores, fluency2)

  plt.scatter(scores, fluency3)
  plt.xlabel("Score")
  plt.ylabel("Concurrent Activity")
  plt.title("Concurrent Activity: Forced Coordination", fontsize="20")

  plt.savefig('fluency3.png')
  stats.pearsonr(scores, fluency3)

