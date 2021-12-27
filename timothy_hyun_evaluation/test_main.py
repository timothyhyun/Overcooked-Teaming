# Timothy Hyun
# December 26, 2021


import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import json
import ast


#import fluency files
import fluency_metrics
import handoff_time
import execution_delay


def main():
    df = pd.read_csv("json.csv")
    df["state"] = df["state"].map(ast.literal_eval)
    df["next_state"] = df["next_state"].map(ast.literal_eval)
    fluency1,fluency2,fluency3 = fluency_metrics.fluencyMeasures(df)
    #list of lists
    handoff = handoff_time.calcHandoff(df)
    execution = execution_delay.executionDelay(df)

    # create csv of results
    rounds = df.groupby("layout_name")
    totals = []

    for i in range(len(df["layout_name"].unique())):
        t = rounds.get_group(df["layout_name"].unique()[i])
        temp = t.groupby("workerid_num")
        hold = df[df["layout_name"] == df["layout_name"].unique()[i]]["workerid_num"].unique()
        for j in range(len(hold)):
            res = temp.get_group(hold[j])
            instance = []
            instance.append(list(res["layout_name"])[0])
            instance.append(list(res["workerid_num"])[0])
            instance.append(max(list(res["score"])))
            instance.append(fluency1[i][j])
            instance.append(fluency2[i][j])
            instance.append(fluency3[i][j])
            instance.append(handoff[i][j])
            instance.append(execution[i][j])
            totals.append(instance)
    res = pd.DataFrame(totals, columns=["layout", "workerid", "score", "human idle time", "robot idle time", "concurrent activity", "handoff time", "execution delay"])
    res.to_csv("metrics.csv")







if __name__ == '__main__':
  main()