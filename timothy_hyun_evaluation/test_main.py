# Timothy Hyun
# December 26, 2021


import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

    means = []
    errors = []
    size = len(fluency1[0])
    means.append([np.mean(fluency1[0]), np.mean(fluency1[1]), np.mean(fluency1[2]), np.mean(fluency1[3]), np.mean(fluency1[4])])
    means.append([np.mean(fluency2[0]), np.mean(fluency2[1]), np.mean(fluency2[2]), np.mean(fluency2[3]), np.mean(fluency2[4])])
    means.append([np.mean(fluency3[0]), np.mean(fluency3[1]), np.mean(fluency3[2]), np.mean(fluency3[3]), np.mean(fluency3[4])])
    means.append([np.mean(handoff[0]), np.mean(handoff[1]), np.mean(handoff[2]), np.mean(handoff[3]), np.mean(handoff[4])])
    means.append([np.mean(execution[0]), np.mean(execution[1]), np.mean(execution[2]), np.mean(execution[3]), np.mean(execution[4])])

    errors.append([np.std(fluency1[0])/ size, np.std(fluency1[1])/ size, np.std(fluency1[2])/ size, np.std(fluency1[3])/ size, np.std(fluency1[4])/ size])
    errors.append([np.std(fluency2[0])/ size, np.std(fluency2[1])/ size, np.std(fluency2[2])/ size, np.std(fluency2[3])/ size, np.std(fluency2[4])/ size])
    errors.append([np.std(fluency3[0])/ size, np.std(fluency3[1])/ size, np.std(fluency3[2])/ size, np.std(fluency3[3])/ size, np.std(fluency3[4])/ size])
    errors.append([np.std(handoff[0])/ size, np.std(handoff[1])/ size, np.std(handoff[2])/ size, np.std(handoff[3])/ size, np.std(handoff[4])/ size])
    errors.append([np.std(execution[0])/ size, np.std(execution[1])/ size, np.std(execution[2])/ size, np.std(execution[3])/ size, np.std(execution[4])/ size])
    plt.rcParams["figure.figsize"] = (15,10)
    x = np.arange(5)
    width = .15
    plt.bar(x-.3, means[0], width, yerr=errors[0])
    plt.bar(x-.15, means[1], width, yerr=errors[1])
    plt.bar(x, means[2], width, yerr=errors[2])
    plt.bar(x+.15, means[3], width, yerr=errors[3])
    plt.bar(x+.3, means[4], width, yerr=errors[4])
    plt.xticks(x, df["layout_name"].unique())
    plt.xlabel("Map")
    plt.ylabel("Metrics")
    plt.legend(["Human Idle Time", "Robot Idle Time", "Concurrent Activity", "Handoff Time",  "Execution Delay"])
    plt.title("Fluency Metrics of Human-AI Data")
    plt.savefig("Human-AiMetrics.png")
    plt.show()
    



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




    # graph based on agent
    bc = df[df["agent_type"] == "ppo_bc"]
    nbc = df[df["agent_type"] != "ppo_bc"]
    #bc metrics
    fluency1bc,fluency2bc,fluency3bc = fluency_metrics.fluencyMeasures(bc)
    #list of lists
    handoffbc = handoff_time.calcHandoff(bc)
    executionbc = execution_delay.executionDelay(bc)
    #adapt metrics
    fluency1nbc,fluency2nbc,fluency3nbc = fluency_metrics.fluencyMeasures(nbc)
    #list of lists
    handoffnbc = handoff_time.calcHandoff(nbc)
    executionnbc = execution_delay.executionDelay(nbc)
    plt.rcParams["figure.figsize"] = (25,10)


    meansbc = []
    meansnbc = []
    meansbc.append([np.mean(fluency1bc[0]), np.mean(fluency1bc[1]), np.mean(fluency1bc[2]), np.mean(fluency1bc[3]), np.mean(fluency1bc[4])])
    meansbc.append([np.mean(fluency2bc[0]), np.mean(fluency2bc[1]), np.mean(fluency2bc[2]), np.mean(fluency2bc[3]), np.mean(fluency2bc[4])])
    meansbc.append([np.mean(fluency3bc[0]), np.mean(fluency3bc[1]), np.mean(fluency3bc[2]), np.mean(fluency3bc[3]), np.mean(fluency3bc[4])])
    meansbc.append([np.mean(handoffbc[0]), np.mean(handoffbc[1]), np.mean(handoffbc[2]), np.mean(handoffbc[3]), np.mean(handoffbc[4])])
    meansbc.append([np.mean(executionbc[0]), np.mean(executionbc[1]), np.mean(executionbc[2]), np.mean(executionbc[3]), np.mean(executionbc[4])])

    meansnbc.append([np.mean(fluency1nbc[0]), np.mean(fluency1nbc[1]), np.mean(fluency1nbc[2]), np.mean(fluency1nbc[3]), np.mean(fluency1nbc[4])])
    meansnbc.append([np.mean(fluency2nbc[0]), np.mean(fluency2nbc[1]), np.mean(fluency2nbc[2]), np.mean(fluency2nbc[3]), np.mean(fluency2nbc[4])])
    meansnbc.append([np.mean(fluency3nbc[0]), np.mean(fluency3nbc[1]), np.mean(fluency3nbc[2]), np.mean(fluency3nbc[3]), np.mean(fluency3nbc[4])])
    meansnbc.append([np.mean(handoffnbc[0]), np.mean(handoffnbc[1]), np.mean(handoffnbc[2]), np.mean(handoffnbc[3]), np.mean(handoffnbc[4])])
    meansnbc.append([np.mean(executionnbc[0]), np.mean(executionnbc[1]), np.mean(executionnbc[2]), np.mean(executionnbc[3]), np.mean(executionnbc[4])])

    size = len(fluency1bc[0])
    errorsbc = []
    errorsnbc = []
    errorsbc.append([np.std(fluency1bc[0])/ size, np.std(fluency1bc[1])/ size, np.std(fluency1bc[2])/ size, np.std(fluency1bc[3])/ size, np.std(fluency1bc[4])/ size])
    errorsbc.append([np.std(fluency2bc[0])/ size, np.std(fluency2bc[1])/ size, np.std(fluency2bc[2])/ size, np.std(fluency2bc[3])/ size, np.std(fluency2bc[4])/ size])
    errorsbc.append([np.std(fluency3bc[0])/ size, np.std(fluency3bc[1])/ size, np.std(fluency3bc[2])/ size, np.std(fluency3bc[3])/ size, np.std(fluency3bc[4])/ size])
    errorsbc.append([np.std(handoffbc[0])/ size, np.std(handoffbc[1])/ size, np.std(handoffbc[2])/ size, np.std(handoffbc[3])/ size, np.std(handoffbc[4])/ size])
    errorsbc.append([np.std(executionbc[0])/ size, np.std(executionbc[1])/ size, np.std(executionbc[2])/ size, np.std(executionbc[3])/ size, np.std(executionbc[4])/ size])
    size = len(fluency1nbc[0])
    errorsnbc.append([np.std(fluency1nbc[0])/ size, np.std(fluency1nbc[1])/ size, np.std(fluency1nbc[2])/ size, np.std(fluency1nbc[3])/ size, np.std(fluency1nbc[4])/ size])
    errorsnbc.append([np.std(fluency2nbc[0])/ size, np.std(fluency2nbc[1])/ size, np.std(fluency2nbc[2])/ size, np.std(fluency2nbc[3])/ size, np.std(fluency2nbc[4])/ size])
    errorsnbc.append([np.std(fluency3nbc[0])/ size, np.std(fluency3nbc[1])/ size, np.std(fluency3nbc[2])/ size, np.std(fluency3nbc[3])/ size, np.std(fluency3nbc[4])/ size])
    errorsnbc.append([np.std(handoffnbc[0])/ size, np.std(handoffnbc[1])/ size, np.std(handoffnbc[2])/ size, np.std(handoffnbc[3])/ size, np.std(handoffnbc[4])/ size])
    errorsnbc.append([np.std(executionnbc[0])/ size, np.std(executionnbc[1])/ size, np.std(executionnbc[2])/ size, np.std(executionnbc[3])/ size, np.std(executionnbc[4])/ size])


    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    colors = ["silver", "lightsteelblue", "cadetblue", "sandybrown", "coral"]
    x = np.arange(0,10,2)
    width = .15
    plt.bar(x-.75, meansbc[0], width, yerr=errorsbc[0], color = colors[0], edgecolor="black")
    plt.bar(x-.45, meansbc[1], width, yerr=errorsbc[1], color = colors[1], edgecolor="black")
    plt.bar(x-.15, meansbc[2], width, yerr=errorsbc[2], color = colors[2], edgecolor="black")
    plt.bar(x+.15, meansbc[3], width, yerr=errorsbc[3], color = colors[3], edgecolor="black")
    plt.bar(x+.45, meansbc[4], width, yerr=errorsbc[4], color = colors[4], edgecolor="black")

    plt.bar(x-.6, meansnbc[0], width, yerr=errorsnbc[0], color = colors[0], edgecolor="black", hatch="/")
    plt.bar(x-.3, meansnbc[1], width, yerr=errorsnbc[1], color = colors[1], edgecolor="black", hatch="/")
    plt.bar(x, meansnbc[2], width, yerr=errorsnbc[2], color = colors[2], edgecolor="black", hatch="/")
    plt.bar(x+.3, meansnbc[3], width, yerr=errorsnbc[3], color = colors[3], edgecolor="black", hatch="/")
    plt.bar(x+.6, meansnbc[4], width, yerr=errorsnbc[4], color = colors[4], edgecolor="black", hatch="/")

    # for a stacked bar
    # x = np.arange(5)
    """ for i in range(5):
        for j in range(5):
            temp = [0 for x in range(5)]
            etemp = [0 for x in range(5)]
            if (meansbc[i][j] > meansnbc[i][j]):
                temp[j] = meansbc[i][j]
                etemp[j] = meansnbc[i][j]
                plt.bar(x+(-.3 + (.15*i)), temp, width, color = colors[i], edgecolor="black")
                plt.bar(x+(-.3 + (.15*i)), etemp, width, color = colors[i], edgecolor="black", hatch="/")
            else:
                temp[j] = meansbc[i][j]
                etemp[j] = meansnbc[i][j]
                plt.bar(x+(-.3 + (.15*i)), etemp, width, color = colors[i], edgecolor="black", hatch="/")
                plt.bar(x+(-.3 + (.15*i)), temp, width, color = colors[i], edgecolor="black")
    """   
    


    blue = mpatches.Patch(facecolor="silver", edgecolor="black",label="Human Idle Time")
    orange = mpatches.Patch(facecolor="lightsteelblue",edgecolor="black", label="Robot Idle Time")
    green = mpatches.Patch(facecolor="cadetblue", edgecolor="black",label="Concurrent Activity")
    red = mpatches.Patch(facecolor="sandybrown", edgecolor="black",label="Handoff Time")
    purple = mpatches.Patch(facecolor="coral",edgecolor="black", label="Execution Delay")
    bc = mpatches.Patch(facecolor="white", edgecolor="black", label="BC Agent")
    nbc = mpatches.Patch(facecolor="white", edgecolor="black", hatch="/", label="Adapt Agent")

    plt.legend(handles=[blue,orange,green,red,purple,bc,nbc], loc=1)

    plt.xticks(x, df["layout_name"].unique())
    plt.xlabel("Map")
    plt.ylabel("Metrics")
    plt.title("Flunency Metrics of different AI agents")
    plt.savefig("Human-AiMetrics-Stacked.png")
    plt.show()






if __name__ == '__main__':
  main()
