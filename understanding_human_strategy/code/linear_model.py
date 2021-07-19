from dependencies import *
from extract_features import run_feature_extraction

def generate_orders_dataset_for_linear():
    team_order_features_dict, _ = run_feature_extraction()
    ## Generate dataset
    # X = order features
    # Y = order rate
    X_data_dict = {}
    Y_data = []
    for team_num in team_order_features_dict:
        team_data = team_order_features_dict[team_num]
        for feature_key in team_data:
            if feature_key == 'other_pot_states':
                continue
            if feature_key == 'order_completion_times':
                Y_data.extend(team_data[feature_key])
            else:
                if feature_key not in X_data_dict:
                    X_data_dict[feature_key] = []
                X_data_dict[feature_key].extend(team_data[feature_key])

    X_data = []
    for feature_key in X_data_dict:
        X_data.append(X_data_dict[feature_key])
    X_data = np.array(X_data).T
    Y_data = np.array(Y_data)

    # print(X_data.shape)
    # print(Y_data.shape)
    return X_data, Y_data


def generate_teams_dataset_for_linear():
    team_order_features_dict, team_num_to_score = run_feature_extraction()
    ## Generate dataset
    # X = team features
    # Y = team score

    ## Generate dataset
    # X = order features
    # Y = order rate
    X_data = []
    Y_data = []
    for team_num in team_order_features_dict:
        team_data = team_order_features_dict[team_num]
        team_add = []
        for feature_key in team_data:
            if feature_key == 'other_pot_states':
                continue
            else:
                #             print('feature_key', feature_key)
                lst = team_data[feature_key]
                feature_mode = max(set(lst), key=lst.count)
                feature_mean = np.mean(lst)
                team_add.append(feature_mode)
                team_add.append(feature_mean)
        X_data.append(team_add)
        Y_data.append(team_num_to_score[team_num])

    X_data = np.array(X_data)
    Y_data = np.array(Y_data)

    # print(X_data.shape)
    # print(Y_data.shape)
    return X_data, Y_data

def run_orders_linear_model():
    X_data, Y_data = generate_orders_dataset_for_linear()

    reg = LinearRegression().fit(X_data, Y_data)
    score = reg.score(X_data, Y_data)
    print('Score', score)
    print('coefficients', reg.coef_)
    print('intercept', reg.intercept_)

def run_teams_linear_model():
    X_data, Y_data = generate_teams_dataset_for_linear()
    reg = LinearRegression().fit(X_data, Y_data)
    score = reg.score(X_data, Y_data)
    print('Score', score)
    print('coefficients', reg.coef_)
    print('intercept', reg.intercept_)



if __name__ == '__main__':
    run_orders_linear_model()
    run_teams_linear_model()








