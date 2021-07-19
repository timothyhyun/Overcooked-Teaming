from dependencies import *
from extract_features import run_feature_extraction

def get_naive_strategy_from_order_features(orders_features_dict):
    strategy_list = []
    num_onions_list = orders_features_dict['other_pot_contains_num_onions']
    for i in range(len(num_onions_list)):
        if num_onions_list[i] == 3:
            strat_num = 1
        elif num_onions_list[i] == 0:
            if i == 0:
                strat_num = 2
            else:
                if num_onions_list[i - 1] == 3:
                    strat_num = 1
                else:
                    strat_num = 2
        else:
            strat_num = 2
        strategy_list.append(strat_num)

    return strategy_list

def get_teams_naive_strategy():
    team_order_features_dict, _ = run_feature_extraction()

    team_order_strats_dict = {}

    for team_num in team_order_features_dict:
        orders_features_dict = team_order_features_dict[team_num]
        team_strat_data = get_naive_strategy_from_order_features(orders_features_dict)

        team_order_strats_dict[team_num] = team_strat_data

        team_order_features_dict[team_num]['strategy_of_order'] = team_strat_data
        plt.scatter(range(len(team_strat_data)), team_strat_data)
        plt.plot(range(len(team_strat_data)), team_strat_data)
        plt.title('Team Number: ' + str(team_num))
        plt.ylabel("Strategy Number")
        plt.xlabel("Order Number")
        plt.savefig('../images/naive_strat_team_'+str(team_num)+'.png')
        plt.close()







