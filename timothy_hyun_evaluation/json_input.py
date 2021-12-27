
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import json


def main(path):
  f = open(path)
  # /content/drive/MyDrive/HARP LAB/overcooked_json.json
  data = json.load(f)


  # needed items: 
  # create dataframe from list of lists
  # time_elapsed, state, next_state, score, workerid+num
  res = []
  temp = []
  count = 0
  counter = 0
  for i in data: 
    
    tempData = i['datastring']
    try: 
      jsonD = json.loads(tempData)["data"]
    except:
      pass
    for j in range(1,len(jsonD)):
      t = []
      td = jsonD[j]["trialdata"]
      try:
        t.append(td["time_elapsed"])
        t.append(td["state"])
        
        t.append(td["next_state"])
        t.append(td["score"])
        if (td["agent_type"] == "ppo_bc"):
          t.append(i["workerid"] + "bc" + str(counter))
        else:
          t.append(i["workerid"] + "adapt" + str(counter))
        t.append(td["round_num"])
        t.append(td["layout_name"])
        
        t.append(td["agent_type"])

        t.append(0)
        t.append(td["joint_action"])
        if (td["round_type"] == "main"):
          temp.append(t)
      
        try:
          jsonD[j+1]["trialdata"]["time_elapsed"]
        except:
          counter += 1
        
      except: 
        count += 1
      

    tempDataFrame = pd.DataFrame(temp, columns=['time_elapsed', 'state', 'next_state', 'score', 'workerid_num', 'round_num', "layout_name", "agent_type", "player_index", "joint_action"])


    
    


  tempDataFrame.to_csv("json.csv")


if __name__ == '__main__':
  main()