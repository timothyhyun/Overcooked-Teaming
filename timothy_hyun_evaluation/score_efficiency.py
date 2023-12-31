
# Timothy Hyun
# November 5, 2021

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({
   'font.size': 12
})




def graohScores(df):
  # .diff first row is empty
  # 205 
  df["score_diff"] = df["score"].diff()
  increase = df[df["score_diff"] != 0]
  increase["round_diff"] = increase["round_num"].diff()
  increase["time_diff"] = increase["time_elapsed"].diff()
  increase = increase[increase["time_diff"] > 0]

  times = increase.groupby("round_num")
  time_diffs = [[]]
  for i in range(41):
    time_diffs.append([])

  for i in range(5):
    t = times.get_group(i)
    temp = t.groupby("workerid_num")
    hold = increase[increase["round_num"] == i]["workerid_num"].unique()
    for j in hold:
      workTemp = temp.get_group(j)
      for k in range(len(workTemp["time_diff"])):
        listTemp = list(workTemp["time_diff"])
        time_diffs[k].append(listTemp[k])

  mean_diffs = []
  std_diffs = []
  for i in time_diffs:
    means = np.mean(i)
    if means < 30:
      mean_diffs.append(means)
    stds = np.std(i)
    if stds < 30:
      std_diffs.append(stds)
  mean_diffs
  std_diffs
  len(mean_diffs)

  plt.plot(mean_diffs)
  lowerend = [a_i - b_i for a_i, b_i in zip(mean_diffs, std_diffs)]
  upperend = [a_i + b_i for a_i, b_i in zip(mean_diffs, std_diffs)]
  plt.xlabel("Score")
  plt.ylabel("Time per Dish Served")
  plt.title("Human Aggregate Score Efficiency", fontsize=20)
  plt.fill_between(range(41), lowerend, upperend, alpha=.5)
  plt.savefig('aggregatescore.png')


listMaxes = [23,41,21,17,24]

def specificRound(i, df): 
  df["score_diff"] = df["score"].diff()
  increase = df[df["score_diff"] != 0]
  increase["round_diff"] = increase["round_num"].diff()
  increase["time_diff"] = increase["time_elapsed"].diff()
  increase = increase[increase["time_diff"] > 0]
  times = increase.groupby("round_num")
  res = [[]]
  for l in range(10):
    res.append([])
  t = times.get_group(i)
  temp = t.groupby("workerid_num")
  hold = increase[increase["round_num"] == i]["workerid_num"].unique()
  for j in hold:
    workTemp = temp.get_group(j)
    for k in range(len(workTemp["time_diff"])):
      listTemp = list(workTemp["time_diff"])
      res[k].append(listTemp[k])
  return res

def specificScore(round):
  roundNum = round
  x = specificRound(roundNum)
  mean_diffs = []
  std_diffs = []
  for i in x:
    means = np.mean(i)
    if means < 30:
      mean_diffs.append(means)
    stds = np.std(i)
    if stds < 30:
      std_diffs.append(stds)

  plt.plot(mean_diffs)
  plt.xlabel("Score")
  plt.ylabel("Time per Dish Served")
  plt.title("AI Score Efficiency: Asymmetric Advantages", fontsize=20)
  lowerend = [a_i - b_i for a_i, b_i in zip(mean_diffs, std_diffs)]
  upperend = [a_i + b_i for a_i, b_i in zip(mean_diffs, std_diffs)]
  plt.fill_between(range(listMaxes[round]), lowerend, upperend, alpha=.5)
  plt.savefig('human4score.png')
