
# Timothy Hyun
# October 29, 2021
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#/content/drive/MyDrive/HARP LAB/clean_main_trials.pkl

def scoreStats(df):
  #df.set_index('Unnamed: 0')
  #df.sort_values(by=['Unnamed: 0'], ascending=True);
  df.to_csv("hadata.csv")

  df["round_diff"] = df["round_num"].diff()
  df["round_diff"] = df['round_diff'].shift(-1)
  starts = df[df["round_diff"] != 0]

  rounds = starts.groupby("round_num")
  round_zero = rounds.get_group(0)
  round_one = rounds.get_group(1)
  round_two = rounds.get_group(2)
  round_three = rounds.get_group(3)
  round_four = rounds.get_group (4)
  # round zero: cramped
  # round one: asymmetric advantages
  # round two: coordination ring
  # round three: random3
  # round four: random0
  print(round_zero['score'].mean(), round_one['score'].mean(), round_two['score'].mean(), 
        round_three['score'].mean(), round_four['score'].mean())

  """

  ```
  # This is formatted as code
  ```



  *   Human-AI: 15.908, 2.698, 4.671, 3.930, 0.798
  *   Human-Human: 74.0, 114.0, 62.5, 47.75, 51.0


  """



  # .diff first row is empty 
  df["score_diff"] = df["score"].diff()
  increase = df[df["score_diff"] > 0]
  increase["round_diff"] = increase["round_num"].diff()
  increase["time_diff"] = increase["time_elapsed"].diff()
  increase["time_shape"] = increase["time_diff"].diff()
  increase["time_diff"][increase["round_diff"] != 0] = 12345678
  increase["time_shape"][increase["time_diff"] == 12345678] = None
  increase["temp"] = increase['time_diff'].shift(1)
  increase["time_shape"][increase["temp"] == 12345678] = None
  increase["time_diff"][increase["round_diff"] != 0] = None
  increase.drop(columns=['temp', 'round_diff'])
  increase.to_csv("final.csv")

  times = increase.groupby("round_num");
  t = times.get_group(0)
  increase[increase["round_num"] == 0]["workerid_num"].unique()

  t.to_csv("t.csv")

  tt = t.groupby("workerid_num")
  asdf = tt.get_group(9)
  asdf["time_shape"].mean()

  # divide by round number:
  roundAnalysis = []
  for i in range (5):
    average = 0
    temp = times.get_group(i)
    hold = increase[increase["round_num"] == i]["workerid_num"].unique()
    for j in hold:
      valid = len(hold)
      av = 0
      tempNum = temp.groupby("workerid_num")
      res = tempNum.get_group(j)
      av = res["time_shape"].mean()
      if (abs(av) <180):
        average = average + av
      else:
        av = 0
        valid -= 1
        average = average + av
    average = average/valid
    roundAnalysis.append(average)
  roundAnalysis

  """

  *   Human-AI: -0.374, -0.141, 0.115, -0.167, -0.073
  *   Human-Human: -0.172, -0.098, -0.165, -0.209, -0.348

  """
