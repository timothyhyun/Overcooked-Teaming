from dependencies import *
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, PlayerState, ObjectState, OvercookedState
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.mdp.actions import Action, Direction
import pdb


from overcooked_ai_py.agents.fixed_strategy_agent import DualPotAgent, FixedStrategy_AgentPair, SinglePotAgent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator

#
# Action Set (For P1/P2)
# 1. North
# 2. South
# 3. East
# 4. West
# 5. Stay
# 6. Pickup onion from dispenser
# 7. Pickup dish from dispenser
# 8. Put down onion on counter
# 9. Put down onion in pot
# 10. Put down dish on counter
# 11. Put down soup on counter
# 12. Pickup soup with dish
# 13. Serve Soup

NORTH = 0
SOUTH = 1
EAST = 3
WEST = 4
STAY_IN_PLACE = 5
PICKUP_ONION_FROM_DISPENSER = 6
PICKUP_DISH_FROM_DISPENSER = 7
PUT_DOWN_ONION_ON_COUNTER = 8
PUT_DOWN_ONION_IN_POT = 9
PUT_DOWN_DISH_ON_COUNTER = 10
PUT_DOWN_SOUP_ON_COUNTER = 11
PICKUP_ONION_FROM_COUNTER = 12
PICKUP_DISH_FROM_COUNTER = 13
PICKUP_SOUP_FROM_COUNTER = 14
PICKUP_SOUP_W_DISH = 15
SERVE_SOUP = 16

N_HIGH_LEVEL_ACTIONS = 17
ACTION_DIRECTION_TO_HIGH_LEVEL = {Direction.NORTH:NORTH, Direction.SOUTH:SOUTH, Direction.EAST:EAST, Direction.WEST:WEST}
ALL_HIGH_LEVEL_ACTIONS = [NORTH, SOUTH, EAST, WEST, STAY_IN_PLACE, PICKUP_ONION_FROM_DISPENSER,
                          PICKUP_DISH_FROM_DISPENSER,
                          PUT_DOWN_ONION_ON_COUNTER, PUT_DOWN_ONION_IN_POT, PUT_DOWN_DISH_ON_COUNTER, PUT_DOWN_SOUP_ON_COUNTER,
                          PICKUP_ONION_FROM_COUNTER, PICKUP_DISH_FROM_COUNTER, PICKUP_SOUP_FROM_COUNTER,
                          PICKUP_SOUP_W_DISH, SERVE_SOUP]

HL_ACTION_IDX_TO_ACTION = {i:x for i,x in enumerate(ALL_HIGH_LEVEL_ACTIONS)}
HL_ACTION_TO_ACTION_IDX = {x:i for i,x in enumerate(ALL_HIGH_LEVEL_ACTIONS)}


N_UNIQUE_STATES = 4672
#
# State features
#
#
# 1. P1 position
# 2. P1 orientation
# 3. P2 position
# 4. P2 orientation
# 5. P1 carrying
# 6. P2 carrying
# 7. Counter 1 state - none, onion, dish, soup
# 8. Counter 2 state - none, onion, dish, soup
# 9. Counter 3 state - none, onion, dish, soup
# 10. Counter 4 state - none, onion, dish, soup
# 11. Counter 5 state - none, onion, dish, soup
# 12. Counter 6 state - none, onion, dish, soup
# 13. Counter 7 state - none, onion, dish, soup
# 14. Top Pot 1 state - empty, 1 onion, 2 onion, 3 onion cooking, 3 onion ready
# 15. Right Pot 2 state - empty, 1 onion, 2 onion, 3 onion cooking, 3 onion ready
# 16. North is wall?
# 17. South is wall?
# 18. East is wall?
# 19. West is wall?
# 20. Facing direction wall is empty counter?


from dependencies import *
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, PlayerState, ObjectState, OvercookedState
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.mdp.actions import Action, Direction

def json_eval(s):
    json_acceptable_string = s.replace("'", "\"")
    d = json.loads(json_acceptable_string)
    return d


def run_one_game_fixed_agents(a0_type, a1_type):
    layout_name = 'random0'

    simple_mdp = OvercookedGridworld.from_layout_name(layout_name, start_order_list=['any'], cook_time=20)
    base_params_start_or = {
        'start_orientations': True,
        'wait_allowed': False,
        'counter_goals': simple_mdp.terrain_pos_dict['X'],
        'counter_drop': [],
        'counter_pickup': [],
        'same_motion_goals': False
    }
    mlp = MediumLevelPlanner(simple_mdp, base_params_start_or)
    # random_agent_0 = CoupledPlanningAgent(mlp)
    # random_agent_1 = CoupledPlanningAgent(mlp)

    random_agent_0 = DualPotAgent(simple_mdp, player_index=0)
    random_agent_1 = DualPotAgent(simple_mdp, player_index=1)
    if a0_type == 'SP':
        random_agent_0 = SinglePotAgent(simple_mdp, player_index=0)
    if a1_type == 'SP':
        random_agent_1 = SinglePotAgent(simple_mdp, player_index=1)



    num_rounds = 1
    display = False
    bc_params = {
                'data_params': {
                    'train_mdps': ['random0'],
                    'ordered_trajs': True,
                    'human_ai_trajs': False,
                    'data_path': '../data/human/anonymized/clean_train_trials.pkl'
                },
                 'mdp_params': {
                     'layout_name': 'random0',
                     'start_order_list': None},
                 'env_params': {'horizon': 400},
                 'mdp_fn_params': {}}

    evaluator = AgentEvaluator(mdp_params=bc_params["mdp_params"], env_params=bc_params["env_params"])

    random_results = evaluator.evaluate_agent_pair(
        FixedStrategy_AgentPair(random_agent_0, random_agent_1, allow_duplicate_agents=False),
        num_games=max(int(num_rounds / 2), 1), display=display)


    avg_random_results = np.mean(random_results['ep_returns'])

    # print('avg', avg_random_results)
    return random_results, avg_random_results




def run_fixed_featurization(a0_type, a1_type):
    name = 'random0'

    # old_trials = import_2019_data()
    # layout_trials = old_trials[old_trials['layout_name'] == name]['trial_id'].unique()

    trial_hl_actions_seq = []

    trial_state_seq = []
    trial_action_seq = []
    trial_feature_seq = []
    trial_sparse_reward_seq = []
    all_states = []

    # print("layout_trials", layout_trials)

    # for j in range(len(layout_trials)):
    # for j in [8]:
    # for trial_id in [114]:
    #     # trial_id = layout_trials[j]
    #     trial_df = old_trials[old_trials['trial_id'] == trial_id]
    #
    #     score = old_trials[old_trials['trial_id'] == trial_id]['score'].to_numpy()[-1]
    #     state_data = trial_df['state'].values
    #     joint_actions = trial_df['joint_action'].values
    #     time_elapsed = trial_df['time_elapsed'].values
    for trial_id in [0]:
        fixed_results, avg_fixed_results = run_one_game_fixed_agents(a0_type, a1_type)
        joint_actions = fixed_results['ep_actions'][0]
        state_data = fixed_results['ep_observations'][0]

        # oc_state = OvercookedState()
        player_idx = 0

        state_seq = []
        action_seq = []
        feature_seq = []
        sparse_reward_seq = []

        hl_action_seq = []
        all_possible_joint_actions = []


        overcooked_mdp = OvercookedGridworld.from_layout_name('random0', start_order_list=['any'], cook_time=20)
        base_params_start_or = {
            'start_orientations': True,
            'wait_allowed': False,
            'counter_goals': overcooked_mdp.terrain_pos_dict['X'],
            'counter_drop': [],
            'counter_pickup': [],
            'same_motion_goals': False
        }
        mlp = MediumLevelPlanner(overcooked_mdp, base_params_start_or)

        for state_i in range(1, len(state_data)):
            # print("json_eval(state_data[state_i])", json_eval(state_data[state_i]))
            # overcooked_state_i = OvercookedState.from_dict(json_eval(state_data[state_i]))
            overcooked_state_i = state_data[state_i]
            # featurized_state_i = featurize_state(overcooked_state_i)
            prev_joint_action_eval = joint_actions[state_i-1]
            prev_joint_action = []
            for elem in prev_joint_action_eval:
                if tuple(elem) !=  ('I', 'N', 'T', 'E', 'R', 'A', 'C', 'T') and tuple(elem) !=  ('i', 'n', 't', 'e', 'r', 'a', 'c', 't'):
                    prev_joint_action.append(tuple(elem))
                else:
                    prev_joint_action.append("interact")


            joint_action_eval = joint_actions[state_i]
            joint_action = []
            for elem in joint_action_eval:
                if tuple(elem) != ('I', 'N', 'T', 'E', 'R', 'A', 'C', 'T') and tuple(elem) !=  ('i', 'n', 't', 'e', 'r', 'a', 'c', 't'):
                    joint_action.append(tuple(elem))
                else:
                    joint_action.append("interact")
            action_label = Action.ACTION_TO_INDEX[joint_action[player_idx]]
            action_label = np.eye(6)[action_label]

            # print("prev_joint_action", prev_joint_action)
            # print('overcooked_state_i', overcooked_state_i)
            team_features = overcooked_mdp.featurize_state_for_irl(overcooked_state_i, mlp, prev_joint_action)

            player_idx_to_high_level_action, reward_featurized_state, sparse_reward = overcooked_mdp.get_high_level_interact_action(overcooked_state_i, joint_action)
            high_level_action_p0, high_level_action_p1 = player_idx_to_high_level_action[0], player_idx_to_high_level_action[1]
            joint_action_indices = (high_level_action_p0, high_level_action_p1)
            # hl_actions_list.append(high_level_action)
            high_level_action_label = np.concatenate([np.eye(N_HIGH_LEVEL_ACTIONS)[high_level_action_p0], np.eye(N_HIGH_LEVEL_ACTIONS)[high_level_action_p1]])
            # print("featurized_state", ordered_features_p0)
            state_seq.append(team_features)
            action_seq.append(high_level_action_label)
            feature_seq.append(reward_featurized_state)
            all_states.append(team_features)
            hl_action_seq.append(joint_action_indices)
            sparse_reward_seq.append(sparse_reward)
            all_possible_joint_actions.append(tuple(joint_action_indices))

        trial_state_seq.append(state_seq)
        trial_action_seq.append(action_seq)
        trial_feature_seq.append(feature_seq)
        trial_hl_actions_seq.append(hl_action_seq)
        trial_sparse_reward_seq.append(sparse_reward_seq)
        break

    unique_states = np.unique(all_states, axis=0)
    n_unique_states = len(unique_states)
    state_idx_to_state = {idx:state for idx,state in enumerate(unique_states)}
    state_tuple_to_state_idx = {tuple(state): idx for idx, state in enumerate(unique_states)}
    # state_tuple_to_state_idx = {tuple(state_idx_to_state[idx]):idx for idx, in state_idx_to_state}

    # unique_features = np.unique(all_states, axis=0)
    # n_unique_states = len(unique_states)
    # state_idx_to_state = {idx: state for idx, state in enumerate(unique_states)}
    # state_tuple_to_state_idx = {tuple(state): idx for idx, state in enumerate(unique_states)}
    all_possible_joint_actions.append((HL_ACTION_TO_ACTION_IDX[STAY_IN_PLACE], HL_ACTION_TO_ACTION_IDX[STAY_IN_PLACE]))
    unique_joint_actions = np.unique(all_possible_joint_actions, axis=0)
    unique_joint_actions = [tuple(x) for x in unique_joint_actions]
    print("unique_joint_actions", unique_joint_actions)
    # n_unique_joint_actions = len(unique_joint_actions)

    # unique_joint_actions = []
    # for action_p0 in ALL_HIGH_LEVEL_ACTIONS:
    #     for action_p1 in ALL_HIGH_LEVEL_ACTIONS:
    #         unique_joint_actions.append((action_p0, action_p1))

    n_unique_joint_actions = len(unique_joint_actions)
    joint_idx_to_action = {idx: act for idx, act in enumerate(unique_joint_actions)}
    joint_action_to_idx = {act: idx for idx, act in enumerate(unique_joint_actions)}

    state_idx_to_reward = {}

    transition_matrix = np.zeros((n_unique_states, n_unique_joint_actions, n_unique_states))

    for idx in range(transition_matrix.shape[0]):
        stay_tuple = (HL_ACTION_TO_ACTION_IDX[STAY_IN_PLACE], HL_ACTION_TO_ACTION_IDX[STAY_IN_PLACE])
        stay_idx = joint_action_to_idx[stay_tuple]
        transition_matrix[idx, stay_idx, idx] = 1
        state_idx_to_reward[idx] = 0


    trajectories = []

    state_to_feature = {}

    for trial_idx in range(len(trial_state_seq)):
        add_to_trajectory = []
        state_list = trial_state_seq[trial_idx]
        act_list = trial_hl_actions_seq[trial_idx]
        rew_list = trial_sparse_reward_seq[trial_idx]
        feature_seq = trial_feature_seq[trial_idx]

        for i in range(len(state_list)-1):
            s = state_tuple_to_state_idx[tuple(state_list[i])]
            a = joint_action_to_idx[act_list[i]]
            add_to_trajectory.append(np.array([s,a]))
            sp = state_tuple_to_state_idx[tuple(state_list[i+1])]

            transition_matrix[s, a, sp] = 1
            if rew_list[i] > 0:
                state_idx_to_reward[s] = rew_list[i]

            featurized_state = feature_seq[i]
            state_to_feature[s] = featurized_state

        trajectories.append(np.array(add_to_trajectory))

    state_reward_list = []
    feature_matrix = []
    for s in state_idx_to_reward:
        state_reward_list.append(state_idx_to_reward[s])
        # print("s = ", s)
        # print(state_to_feature)
        if s not in state_to_feature:
            print("not found")
            feature_matrix.append([0]*6)
        else:
            feature_matrix.append(state_to_feature[s])


    # print("unique_states", state_idx_to_state) # 4672
    # print("n_unique_states", n_unique_states)
    trial_state_seq = np.array(trial_state_seq)
    trial_action_seq = np.array(trial_action_seq)
    trial_feature_seq = np.array(trial_feature_seq)
    state_reward_list = np.array(state_reward_list)

    trajectories = np.array(trajectories)
    print("trajectories", trajectories.shape)
    feature_matrix = np.array(feature_matrix)
    # print("state_idx_to_state", state_idx_to_state)
    # print(sum(transition_matrix[0, SOUTH]))



    # pdb.set_trace()



    return trial_state_seq, trial_action_seq, trial_feature_seq, transition_matrix, state_idx_to_state, state_tuple_to_state_idx, state_reward_list, feature_matrix, trajectories


def run_featurization_bc():
    name = 'random0'

    old_trials = import_2019_data()
    layout_trials = old_trials[old_trials['layout_name'] == name]['trial_id'].unique()

    trial_hl_actions_seq = []

    trial_state_seq = []
    trial_action_seq = []
    trial_feature_seq = []
    trial_sparse_reward_seq = []
    all_states = []

    print("layout_trials", layout_trials)
    X = []
    Y = []

    for j in range(len(layout_trials)):
    # for j in [8]:
    # for trial_id in [114]:
        trial_id = layout_trials[j]
        trial_df = old_trials[old_trials['trial_id'] == trial_id]

        score = old_trials[old_trials['trial_id'] == trial_id]['score'].to_numpy()[-1]
        state_data = trial_df['state'].values
        joint_actions = trial_df['joint_action'].values
        time_elapsed = trial_df['time_elapsed'].values

        # oc_state = OvercookedState()
        player_idx = 0

        state_seq = []
        action_seq = []
        feature_seq = []
        sparse_reward_seq = []

        hl_action_seq = []



        overcooked_mdp = OvercookedGridworld.from_layout_name('random0', start_order_list=['any'], cook_time=20)
        base_params_start_or = {
            'start_orientations': True,
            'wait_allowed': False,
            'counter_goals': overcooked_mdp.terrain_pos_dict['X'],
            'counter_drop': [],
            'counter_pickup': [],
            'same_motion_goals': False
        }
        mlp = MediumLevelPlanner(overcooked_mdp, base_params_start_or)

        for state_i in range(1, len(state_data)):
            # print("json_eval(state_data[state_i])", json_eval(state_data[state_i]))
            overcooked_state_i = OvercookedState.from_dict(json_eval(state_data[state_i]))
            # featurized_state_i = featurize_state(overcooked_state_i)
            prev_joint_action_eval = json_eval(joint_actions[state_i-1])
            prev_joint_action = []
            for elem in prev_joint_action_eval:
                if tuple(elem) !=  ('I', 'N', 'T', 'E', 'R', 'A', 'C', 'T'):
                    prev_joint_action.append(tuple(elem))
                else:
                    prev_joint_action.append("interact")


            joint_action_eval = json_eval(joint_actions[state_i])
            joint_action = []
            for elem in joint_action_eval:
                if tuple(elem) != ('I', 'N', 'T', 'E', 'R', 'A', 'C', 'T'):
                    joint_action.append(tuple(elem))
                else:
                    joint_action.append("interact")
            action_label = Action.ACTION_TO_INDEX[joint_action[player_idx]]
            action_label = np.eye(6)[action_label]

            # print("prev_joint_action", prev_joint_action)
            # print('overcooked_state_i', overcooked_state_i)
            team_features = overcooked_mdp.featurize_state_for_irl(overcooked_state_i, mlp, prev_joint_action)

            player_idx_to_high_level_action, reward_featurized_state, sparse_reward = overcooked_mdp.get_high_level_interact_action(overcooked_state_i, joint_action)
            high_level_action_p0, high_level_action_p1 = player_idx_to_high_level_action[0], player_idx_to_high_level_action[1]
            joint_action_indices = (high_level_action_p0, high_level_action_p1)
            # hl_actions_list.append(high_level_action)
            high_level_action_label = np.concatenate([np.eye(N_HIGH_LEVEL_ACTIONS)[high_level_action_p0], np.eye(N_HIGH_LEVEL_ACTIONS)[high_level_action_p1]])
            # print("featurized_state", ordered_features_p0)
            state_seq.append(team_features)
            action_seq.append(high_level_action_label)
            feature_seq.append(reward_featurized_state)
            all_states.append(team_features)
            hl_action_seq.append(joint_action_indices)
            sparse_reward_seq.append(sparse_reward)

            X.append(np.concatenate([team_features, reward_featurized_state]))
            Y.append(high_level_action_label)


        trial_state_seq.append(state_seq)
        trial_action_seq.append(action_seq)
        trial_feature_seq.append(feature_seq)
        trial_hl_actions_seq.append(hl_action_seq)
        trial_sparse_reward_seq.append(sparse_reward_seq)
        # break

    unique_states = np.unique(all_states, axis=0)
    n_unique_states = len(unique_states)
    state_idx_to_state = {idx:state for idx,state in enumerate(unique_states)}
    state_tuple_to_state_idx = {tuple(state): idx for idx, state in enumerate(unique_states)}
    # state_tuple_to_state_idx = {tuple(state_idx_to_state[idx]):idx for idx, in state_idx_to_state}

    # unique_features = np.unique(all_states, axis=0)
    # n_unique_states = len(unique_states)
    # state_idx_to_state = {idx: state for idx, state in enumerate(unique_states)}
    # state_tuple_to_state_idx = {tuple(state): idx for idx, state in enumerate(unique_states)}

    # unique_joint_actions = np.unique(joint_action_indices, axis=0)
    # n_unique_joint_actions = len(unique_joint_actions)

    unique_joint_actions = []
    for action_p0 in ALL_HIGH_LEVEL_ACTIONS:
        for action_p1 in ALL_HIGH_LEVEL_ACTIONS:
            unique_joint_actions.append((action_p0, action_p1))

    n_unique_joint_actions = len(unique_joint_actions)
    joint_idx_to_action = {idx: act for idx, act in enumerate(unique_joint_actions)}
    joint_action_to_idx = {act: idx for idx, act in enumerate(unique_joint_actions)}

    state_idx_to_reward = {}

    transition_matrix = np.zeros((n_unique_states, n_unique_joint_actions, n_unique_states))

    for idx in range(transition_matrix.shape[0]):
        stay_tuple = (HL_ACTION_TO_ACTION_IDX[STAY_IN_PLACE], HL_ACTION_TO_ACTION_IDX[STAY_IN_PLACE])
        stay_idx = joint_action_to_idx[stay_tuple]
        transition_matrix[idx, stay_idx, idx] = 1
        state_idx_to_reward[idx] = 0


    trajectories = []

    state_to_feature = {}

    for trial_idx in range(len(trial_state_seq)):
        add_to_trajectory = []
        state_list = trial_state_seq[trial_idx]
        act_list = trial_hl_actions_seq[trial_idx]
        rew_list = trial_sparse_reward_seq[trial_idx]
        feature_seq = trial_feature_seq[trial_idx]

        for i in range(len(state_list)-1):
            s = state_tuple_to_state_idx[tuple(state_list[i])]
            a = joint_action_to_idx[act_list[i]]
            add_to_trajectory.append(np.array([s,a]))
            sp = state_tuple_to_state_idx[tuple(state_list[i+1])]

            transition_matrix[s, a, sp] = 1
            if rew_list[i] > 0:
                state_idx_to_reward[s] = rew_list[i]

            featurized_state = feature_seq[i]
            state_to_feature[s] = featurized_state

        trajectories.append(np.array(add_to_trajectory))

    state_reward_list = []
    feature_matrix = []
    for s in state_idx_to_reward:
        state_reward_list.append(state_idx_to_reward[s])
        # print("s = ", s)
        # print(state_to_feature)
        if s not in state_to_feature:
            print("not found")
            feature_matrix.append([0]*6)
        else:
            feature_matrix.append(state_to_feature[s])


    # print("unique_states", state_idx_to_state) # 4672
    # print("n_unique_states", n_unique_states)
    trial_state_seq = np.array(trial_state_seq)
    trial_action_seq = np.array(trial_action_seq)
    trial_feature_seq = np.array(trial_feature_seq)
    state_reward_list = np.array(state_reward_list)

    trajectories = np.array(trajectories)
    print("trajectories", trajectories.shape)
    feature_matrix = np.array(feature_matrix)
    # print("state_idx_to_state", state_idx_to_state)
    # print(sum(transition_matrix[0, SOUTH]))



    # pdb.set_trace()



    # return trial_state_seq, trial_action_seq, trial_feature_seq, transition_matrix, state_idx_to_state, state_tuple_to_state_idx, state_reward_list, feature_matrix, trajectories
    return np.array(X), np.array(Y)

def main():
    state_seq, action_seq, feature_seq, transition_matrix, state_idx_to_state, state_tuple_to_state_idx, state_reward_list, feature_matrix, trajectories = run_featurization()


    print("state_seq shape", state_seq.shape)
    print('action_seq shape', action_seq.shape)
    print("feature_seq.shape = ", feature_seq.shape)
    # print(feature_seq)




if __name__ == '__main__':
    main()
















