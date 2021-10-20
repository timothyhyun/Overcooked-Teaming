import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import defaultdict

from overcooked_ai_py.utils import save_pickle, load_pickle
from overcooked_ai_py.agents.agent import AgentPair, RandomAgent, CoupledPlanningAgent
from overcooked_ai_py.agents.fixed_strategy_agent import DualPotAgent, FixedStrategy_AgentPair, SinglePotAgent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.utils import save_pickle
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, PlayerState, ObjectState, OvercookedState

from understanding_human_strategy.code.dependencies import *
from understanding_human_strategy.code.hmm import supervised_HMM, unsupervised_HMM, HiddenMarkovModel
from understanding_human_strategy.code.extract_features import *

from understanding_human_strategy.code.cluster_leven import perform_clustering_on_int_strings

# DEFINE HMM ACTIONS
## 1. P1 Pass onion
## 2. P1 Pass dish
## 3. P2 Put onion in pot
# 4. P2 Serve soup
# P1_PASS_ONION = 0
# P1_PASS_DISH = 1
P2_PUT_ONION_IN_POT = 0
P2_SERVE_SOUP = 1
P1_PASS_ONION = 2
P1_PASS_DISH = 3



def run_one_game_fixed_agents(a0_type, a1_type):
    layout_name = 'random0'

    simple_mdp = OvercookedGridworld.from_layout_name('random0', start_order_list=['any'], cook_time=20)
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
        random_agent_1 = SinglePotAgent(simple_mdp, player_index=0)



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


def run_ppo_bc_game(layout, num_rounds, bc_model_paths, ppo_bc_model_paths, seeds, best=False,
                                          display=False):
    # evaluate_ppo_and_bc_models_for_layout(layout, num_rounds, best_bc_model_paths, ppo_bc_model_paths, seeds=seeds, best=best)
    assert len(seeds["bc_train"]) == len(seeds["bc_test"])
    ppo_bc_performance = defaultdict(lambda: defaultdict(list))

    agent_bc_test, bc_params = get_bc_agent_from_saved(bc_model_paths['test'][layout])
    ppo_bc_train_path = ppo_bc_model_paths['bc_train'][layout]
    ppo_bc_test_path = ppo_bc_model_paths['bc_test'][layout]
    evaluator = AgentEvaluator(mdp_params=bc_params["mdp_params"], env_params=bc_params["env_params"])

    # num_rounds = 10
    seed_input = 9456
    agent_ppo_bc_train, ppo_config = get_ppo_agent(ppo_bc_train_path, seed_input, best=False)
    assert common_keys_equal(bc_params["mdp_params"], ppo_config["mdp_params"])

    # For curiosity, how well does agent do with itself?
    # ppo_and_ppo = evaluator.evaluate_agent_pair(AgentPair(agent_ppo_bc_train, agent_ppo_bc_train, allow_duplicate_agents=True), num_games=max(int(num_rounds/2), 1), display=display)
    # avg_ppo_and_ppo = np.mean(ppo_and_ppo['ep_returns'])
    # ppo_bc_performance[layout]["PPO_BC_train+PPO_BC_train"].append(avg_ppo_and_ppo)

    # How well it generalizes to new agent in simulation?
    ppo_and_bc = evaluator.evaluate_agent_pair(AgentPair(agent_ppo_bc_train, agent_bc_test), num_games=num_rounds,
                                               display=display)
    avg_ppo_and_bc = np.mean(ppo_and_bc['ep_returns'])
    ppo_bc_performance[layout]["PPO_BC_train+BC_test_0"].append(avg_ppo_and_bc)

    bc_and_ppo = evaluator.evaluate_agent_pair(AgentPair(agent_bc_test, agent_ppo_bc_train), num_games=num_rounds,
                                               display=display)
    avg_bc_and_ppo = np.mean(bc_and_ppo['ep_returns'])
    ppo_bc_performance[layout]["PPO_BC_train+BC_test_1"].append(avg_bc_and_ppo)

    # How well could we do if we knew true model BC_test?
    agent_ppo_bc_test, ppo_config = get_ppo_agent(ppo_bc_test_path, seeds["bc_test"][seed_idx], best=best)
    assert common_keys_equal(bc_params["mdp_params"], ppo_config["mdp_params"])

    ppo_and_bc = evaluator.evaluate_agent_pair(AgentPair(agent_ppo_bc_test, agent_bc_test), num_games=num_rounds,
                                               display=display)
    avg_ppo_and_bc = np.mean(ppo_and_bc['ep_returns'])
    ppo_bc_performance[layout]["PPO_BC_test+BC_test_0"].append(avg_ppo_and_bc)

    bc_and_ppo = evaluator.evaluate_agent_pair(AgentPair(agent_bc_test, agent_ppo_bc_test), num_games=num_rounds,
                                               display=display)
    avg_bc_and_ppo = np.mean(bc_and_ppo['ep_returns'])
    ppo_bc_performance[layout]["PPO_BC_test+BC_test_1"].append(avg_bc_and_ppo)

    return ppo_bc_performance


def track_p2_actions(p1_data, p2_data, objects_data, p1_actions,
                     p2_actions, name, time_elapsed):
    # Save player 1 and 2 actions
    p1_actions_list = []
    p2_actions_list = []

    N_steps = len(p1_data)
    a_min = 1  # the minimial value of the paramater a
    a_max = N_steps - 1  # the maximal value of the paramater a
    a_init = 1  # the value of the parameter a to be used initially, when the graph is created

    t = np.linspace(0, N_steps - 1, N_steps)

    # Define Counters
    counter_location_to_id = {
        (2, 1): 1,
        (2, 2): 2,
        (2, 3): 3,
        (1, 0): 4,
        (1, 4): 5,
        (4, 2): 6,
        (4, 3): 7
    }

    # P2 took what action when one pot had 3 onions? Fill the other or Go for a plate?
    # 0 = P1 passes onion
    # 1 = P1 passes dish
    # 2 = P2 puts onion in pot
    # 3 = P2 serves
    p2_time_strategic_action = []
    p2_time_strategy = []

    both_players_strategic_action = []
    get_next_action = False

    #     p1_private_counters = [4,5]
    #     p2_private_counters = [6,7]
    #     shared_counters = [1,2,3]
    p1_private_counters = [(1, 0), (1, 4)]
    p2_private_counters = [(4, 2), (4, 3)]
    shared_counters = [(2, 1), (2, 2), (2, 3)]

    onion_dispenser_locations = [(0, 1), (0, 2)]
    dish_dispenser_locations = [(0, 3)]

    obj_count_id = 0
    next_obj_count_id = 0

    object_list_tracker = {}
    object_location_tracker = {}

    ordered_delivered_tracker = {}
    top_soup_counter_id = 0  # (3,0)
    right_soup_counter_id = 0  # (4,1)
    top_soup_contents_dict = {'this_contents': [], 'other_contents': [], 'other_state': 'empty'}  # (3,0)
    right_soup_contents_dict = {'this_contents': [], 'other_contents': [], 'other_state': 'empty'}  # (4,1)
    # states: empty, cooking, cooked, partial

    p2_carrying_soup = None
    p2_carrying_soup_pot_side = None
    p2_carrying_soup_pot_side_id = None
    absolute_order_counter = 0
    p2_time_picked_up_soup = None
    p2_time_delivered_soup = None

    players_holding = {1: None, 2: None}

    # layout = eval(old_trials[old_trials['layout_name'] == name]['layout'].to_numpy()[0])
    # layout = np.array([list(elem) for elem in layout])
    # grid_display = np.zeros((layout.shape[0], layout.shape[1], 3))

    p1_major_action = 17  # initialize as doing nothing (stationary)
    p2_major_action = 17

    hmm_actions_list = []

    # print("p1_actions", p1_actions)
    # print("p2_actions", p2_actions)
    # loop over your images
    for a in range(len(t) - 1):

        # for a in range(100):

        p1_x, p1_y = f_p1(t, a, p1_data)[0], f_p1(t, a, p1_data)[1]
        p1_dir_x, p1_dir_y = arrow_p1(t, a, p1_data)[2], arrow_p1(t, a, p1_data)[3]
        p1_obj_x, p1_obj_y, p1_obj_name = obj_p1(t, a, p1_data)[0], \
                                          obj_p1(t, a, p1_data)[1], \
                                          obj_p1(t, a, p1_data)[2]
        p1_act = action_p1(t, a, p1_actions)

        p2_x, p2_y = f_p2(t, a, p1_data)[0], f_p2(t, a, p1_data)[1]
        p2_dir_x, p2_dir_y = arrow_p2(t, a, p2_data)[2], arrow_p2(t, a, p2_data)[3]
        p2_obj_x, p2_obj_y, p2_obj_name = obj_p2(t, a, p2_data)[0], \
                                          obj_p2(t, a, p2_data)[1], \
                                          obj_p2(t, a, p2_data)[2]
        p2_act = action_p2(t, a, p2_actions)
        objects_list = world_obj(t, a, objects_data)
        t1 = time_elapsed[a]

        # cook_state
        # 0 = uncooked
        # 1 = cooking
        # 2 = cooked

        b = a + 1
        p1_x_next, p1_y_next = f_p1(t, b, p1_data)[0], f_p1(t, b, p1_data)[1]
        p1_dir_x_next, p1_dir_y_next = arrow_p1(t, b, p1_data)[2], arrow_p1(t, b, p1_data)[3]
        p1_obj_x_next, p1_obj_y_next, p1_obj_name_next = obj_p1(t, b, p1_data)[0], \
                                                         obj_p1(t, b, p1_data)[1], \
                                                         obj_p1(t, b, p1_data)[2]
        p1_act_next = action_p1(t, b, p1_actions)

        p2_x_next, p2_y_next = f_p2(t, b, p2_data)[0], f_p2(t, b, p2_data)[1]
        p2_dir_x_next, p2_dir_y_next = arrow_p2(t, b, p2_data)[2], arrow_p2(t, b, p2_data)[3]
        p2_obj_x_next, p2_obj_y_next, p2_obj_name_next = obj_p2(t, b, p2_data)[0], \
                                                         obj_p2(t, b, p2_data)[1], \
                                                         obj_p2(t, b, p2_data)[2]
        p2_act_next = action_p2(t, a, p2_actions)
        objects_list_next = world_obj(t, b, objects_data)
        t2 = time_elapsed[b]

        # print('(p1_act, p1_act_next, p2_act, p2_act_next)', (p1_act, p1_act_next, p2_act, p2_act_next))


        ################## PLAYER 1'S MOVEMENT ##################
        # If P1 moves or stays
        if p1_act in ['N', 'S', 'E', 'W']:
            #         print(p1_act, (p1_dir_x_next, p1_dir_y_next))

            # If P1 is carrying something and moving (dish or onion)
            if players_holding[1] is not None:
                obj_location = (p1_obj_x_next, p1_obj_y_next)
                object_held_id = players_holding[1]

                prev_location = object_list_tracker[object_held_id]['location']
                object_list_tracker[object_held_id]['n_actions_since_pickup'] += 1
                object_list_tracker[object_held_id]['location'] = obj_location
                object_list_tracker[object_held_id]['p1_n_actions_since_pickup'] += 1

                object_location_tracker.pop(prev_location, None)
                object_location_tracker[obj_location] = object_held_id

                if p1_obj_name_next == 'onion':
                    p1_major_action = 13
                elif p1_obj_name_next == 'dish':
                    p1_major_action = 14
                elif p1_obj_name_next == 'soup':
                    p1_major_action = 15
            else:
                p1_major_action = 16

        # If P1 interacted
        if p1_act == 'I':
            # If P1 picked up an object
            if p1_obj_x is None and p1_obj_x_next is not None:
                obj_location = (p1_obj_x_next, p1_obj_y_next)
                object_location_tracker[obj_location] = obj_count_id

                if obj_count_id not in object_list_tracker:
                    object_list_tracker[obj_count_id] = {}
                object_list_tracker[obj_count_id]['name'] = p1_obj_name_next
                object_list_tracker[obj_count_id]['player_holding'] = 1
                object_list_tracker[obj_count_id]['id'] = obj_count_id
                object_list_tracker[obj_count_id]['n_actions_since_pickup'] = 0
                object_list_tracker[obj_count_id]['location'] = obj_location
                object_list_tracker[obj_count_id]['on_screen'] = True
                object_list_tracker[obj_count_id]['player_holding_list'] = [1]
                object_list_tracker[obj_count_id]['p1_n_actions_since_pickup'] = 1
                object_list_tracker[obj_count_id]['p2_n_actions_since_pickup'] = 0
                object_list_tracker[obj_count_id]['counter_used'] = []
                object_list_tracker[obj_count_id]['p1_time_started'] = t1
                object_list_tracker[obj_count_id]['p1_time_completed'] = None
                object_list_tracker[obj_count_id]['p2_time_started'] = None
                object_list_tracker[obj_count_id]['p2_time_completed'] = None

                players_holding[1] = obj_count_id

                obj_count_id += 1

                picked_up_from_x, picked_up_from_y = list(p1_x_next)[0] + p1_dir_x_next, list(p1_y_next)[
                    0] + p1_dir_y_next
                picked_up_loc = (picked_up_from_x, picked_up_from_y)

                if p1_obj_name_next == 'onion':
                    if picked_up_loc in onion_dispenser_locations:
                        p1_major_action = 1
                    if picked_up_loc in p1_private_counters:
                        p1_major_action = 20
                    if picked_up_loc in shared_counters:
                        p1_major_action = 18


                elif p1_obj_name_next == 'dish':
                    if picked_up_loc in dish_dispenser_locations:
                        p1_major_action = 2
                    if picked_up_loc in p1_private_counters:
                        p1_major_action = 21
                    if picked_up_loc in shared_counters:
                        p1_major_action = 19

                elif p1_obj_name_next == 'soup':
                    if picked_up_loc in p1_private_counters:
                        p1_major_action = 22
                    if picked_up_loc in shared_counters:
                        p1_major_action = 23

            # If P1 put down an object
            if p1_obj_x is not None and p1_obj_x_next is None:
                object_held_id = players_holding[1]

                object_list_tracker[object_held_id]['player_holding'] = 0
                object_list_tracker[object_held_id]['n_actions_since_pickup'] += 1
                object_list_tracker[object_held_id]['player_holding_list'].append(0)
                object_list_tracker[object_held_id]['p1_n_actions_since_pickup'] += 1

                placed_obj_x, placed_obj_y = list(p1_x_next)[0] + p1_dir_x_next, list(p1_y_next)[0] + p1_dir_y_next
                old_obj_location = object_list_tracker[object_held_id]['location']
                new_obj_location = (placed_obj_x, placed_obj_y)
                object_list_tracker[object_held_id]['location'] = new_obj_location

                object_location_tracker.pop(old_obj_location, None)
                object_location_tracker[new_obj_location] = object_held_id

                counter_index = counter_location_to_id[(placed_obj_x, placed_obj_y)]
                object_list_tracker[object_held_id]['counter_used'].append(counter_index)

                object_list_tracker[object_held_id]['p1_time_completed'] = t2

                players_holding[1] = None

                put_down_loc = new_obj_location
                if p1_obj_name == 'onion':
                    hmm_actions_list.append(P1_PASS_ONION)
                    both_players_strategic_action.append(0)
                    if put_down_loc in p1_private_counters:
                        p1_major_action = 10
                    if put_down_loc in shared_counters:
                        p1_major_action = 3


                elif p1_obj_name == 'dish':
                    hmm_actions_list.append(P1_PASS_DISH)
                    both_players_strategic_action.append(1)
                    if put_down_loc in p1_private_counters:
                        p1_major_action = 11
                    if put_down_loc in shared_counters:
                        p1_major_action = 4

                elif p1_obj_name == 'soup':
                    if put_down_loc in p1_private_counters:
                        p1_major_action = 12
                    if put_down_loc in shared_counters:
                        p1_major_action = 24

        ################## PLAYER 2'S MOVEMENT ##################
        # If P2 moves or stays
        if p2_act in ['N', 'S', 'E', 'W']:

            # If P2 is moving with an object: onion, dish, or soup
            if players_holding[2] is not None:
                obj_location = (p2_obj_x_next, p2_obj_y_next)
                object_held_id = players_holding[2]

                prev_location = object_list_tracker[object_held_id]['location']
                object_list_tracker[object_held_id]['n_actions_since_pickup'] += 1
                object_list_tracker[object_held_id]['location'] = obj_location
                object_list_tracker[object_held_id]['p2_n_actions_since_pickup'] += 1

                object_location_tracker.pop(prev_location, None)
                object_location_tracker[obj_location] = object_held_id

                if p2_obj_name_next == 'onion':
                    p2_major_action = 13
                elif p2_obj_name_next == 'dish':
                    p2_major_action = 14
                elif p2_obj_name_next == 'soup':
                    p2_major_action = 15
            else:
                p2_major_action = 16

        # If P2 interacted
        if p2_act == 'I':
            placed_x, placed_y = list(p2_x_next)[0] + p2_dir_x_next, list(p2_y_next)[0] + p2_dir_y_next
            # If P2 picked up soup from the pot. P2 would be carrying a dish
            if p2_obj_name_next == 'soup':

                # P2 was already carrying this soup
                #                 if p2_obj_name == 'soup':
                #                     print('WHAT IS GOING ON', (p2_data[a], p2_actions[a]))
                #                     print('WHAT IS GOING ON NEXT', (p2_data[b], p2_actions[b]))
                #                     print('picked up from', (placed_x, placed_y))
                #                     print()

                # P2 filled up a dish with the soup
                if p2_obj_name == 'dish':
                    objects_status = objects_data[a]
                    objects_status_next = objects_data[b]
                    top_pot_status = 'empty'
                    right_pot_status = 'empty'
                    for obj_status in objects_status:
                        if obj_status['name'] == 'soup' and obj_status['position'] == [3, 0]:
                            if obj_status['is_cooking'] == True:
                                top_pot_status = 'cooking'
                            if obj_status['is_ready'] == True:
                                top_pot_status = 'ready'
                            if obj_status['is_idle'] == True:
                                top_pot_status = 'idle'
                        if obj_status['name'] == 'soup' and obj_status['position'] == [4, 1]:
                            if obj_status['is_cooking'] == True:
                                right_pot_status = 'cooking'
                            if obj_status['is_ready'] == True:
                                right_pot_status = 'ready'
                            if obj_status['is_idle'] == True:
                                right_pot_status = 'idle'

                    top_soup_contents_dict['other_state'] = right_pot_status
                    right_soup_contents_dict['other_state'] = top_pot_status

                    p2_time_picked_up_soup = t1

                    # picked up from top counter
                    if (placed_x, placed_y) == (3, 0):
                        p2_carrying_soup = copy.deepcopy(top_soup_contents_dict)
                        p2_carrying_soup_pot_side = 'top'
                        p2_carrying_soup_pot_side_id = top_soup_counter_id

                        top_soup_contents_dict['this_contents'] = []  # (3,0)
                        right_soup_contents_dict['other_contents'] = []
                        right_soup_contents_dict['other_state'] = 'empty'
                        top_soup_counter_id += 1

                        p2_major_action = 7

                    #                     objects_data_next [{'name': 'soup', 'position': [3, 0], '_ingredients': [{'name': 'onion', 'position': [3, 0]}], 'cooking_tick': -1, 'is_cooking': False, 'is_ready': False, 'is_idle': True, 'cook_time': -1, '_cooking_tick': -1}]

                    # picked up from right counter
                    if (placed_x, placed_y) == (4, 1):
                        p2_carrying_soup = copy.deepcopy(right_soup_contents_dict)
                        p2_carrying_soup_pot_side = 'right'
                        p2_carrying_soup_pot_side_id = right_soup_counter_id

                        right_soup_contents_dict['this_contents'] = []  # (4,1)
                        top_soup_contents_dict['other_contents'] = []
                        top_soup_contents_dict['other_state'] = 'empty'
                        right_soup_counter_id += 1

                        p2_major_action = 8

            # If P2 picked up an object
            if p2_obj_x is None and p2_obj_x_next is not None:
                placed_x, placed_y = list(p2_x_next)[0] + p2_dir_x_next, list(p2_y_next)[0] + p2_dir_y_next

                if (placed_x, placed_y) in object_location_tracker:
                    obj_picked_id = object_location_tracker[(placed_x, placed_y)]
                else:
                    # print('!!! problem p2 pickup not found')
                    nearest_key = min(list(object_location_tracker.keys()),
                                      key=lambda c: (c[0] - placed_x) ** 2 + (c[1] - placed_y) ** 2)
                    obj_picked_id = object_location_tracker[nearest_key]

                new_obj_location = (p2_obj_x_next, p2_obj_y_next)

                object_list_tracker[obj_picked_id]['player_holding'] = 2
                object_list_tracker[obj_picked_id]['n_actions_since_pickup'] += 1
                object_list_tracker[obj_picked_id]['location'] = new_obj_location
                object_list_tracker[obj_picked_id]['on_screen'] = True
                object_list_tracker[obj_picked_id]['player_holding_list'].append(2)
                object_list_tracker[obj_picked_id]['p2_n_actions_since_pickup'] += 1
                object_list_tracker[obj_picked_id]['p2_time_started'] = t1

                players_holding[2] = obj_picked_id

                object_location_tracker.pop((placed_x, placed_y), None)
                object_location_tracker[new_obj_location] = obj_picked_id

                # new
                picked_up_loc = (placed_x, placed_y)

                if p2_obj_name_next == 'onion':
                    if picked_up_loc in p2_private_counters:
                        p2_major_action = 20
                    if picked_up_loc in shared_counters:
                        p2_major_action = 18


                elif p2_obj_name_next == 'dish':
                    if picked_up_loc in p2_private_counters:
                        p2_major_action = 21
                    if picked_up_loc in shared_counters:
                        p2_major_action = 19

                elif p2_obj_name_next == 'soup':
                    if picked_up_loc in p2_private_counters:
                        p2_major_action = 22
                    if picked_up_loc in shared_counters:
                        p2_major_action = 23

            # If P2 put down an object
            if p2_obj_x is not None and p2_obj_x_next is None:
                # print("players_holding", players_holding)
                object_held_id = players_holding[2]
                # print("object_held_id", object_held_id)

                # print("A1", (p1_act, p1_act_next))
                # print("A2", (p2_act, p2_act_next))
                #
                # print("p2", (p2_obj_x, p2_obj_x_next))

                placed_obj_x, placed_obj_y = list(p2_x_next)[0] + p2_dir_x_next, list(p2_y_next)[0] + p2_dir_y_next

                # If P2 delivered a soup
                if p2_obj_name == 'soup' and (placed_obj_x, placed_obj_y) == (3, 4):
                    hmm_actions_list.append(P2_SERVE_SOUP)
                    p2_time_delivered_soup = t2
                    ordered_delivered_tracker[absolute_order_counter] = {}
                    ordered_delivered_tracker[absolute_order_counter]['details'] = p2_carrying_soup
                    ordered_delivered_tracker[absolute_order_counter]['pot_side'] = p2_carrying_soup_pot_side
                    ordered_delivered_tracker[absolute_order_counter]['pot_side_id'] = p2_carrying_soup_pot_side_id
                    ordered_delivered_tracker[absolute_order_counter]['time_picked_up'] = p2_time_picked_up_soup
                    ordered_delivered_tracker[absolute_order_counter]['time_delivered'] = p2_time_delivered_soup

                    absolute_order_counter += 1
                    p2_major_action = 9

                    p2_time_strategy.append(t1)

                    both_players_strategic_action.append(3)

                    if get_next_action == True:
                        p2_time_strategic_action.append((t1, 1))
                        get_next_action = False


                else:

                    # placed at top counter pot
                    if (placed_obj_x, placed_obj_y) == (3, 0):
                        hmm_actions_list.append(P2_PUT_ONION_IN_POT)
                        top_soup_contents_dict['this_contents'].append(object_held_id)
                        right_soup_contents_dict['other_contents'].append(object_held_id)

                        if p2_obj_name == 'onion':
                            both_players_strategic_action.append(2)
                            p2_major_action = 5

                        if get_next_action == True:
                            p2_time_strategic_action.append((t1, 0))
                            get_next_action = False

                        # if the top pot has 3 onions and right pot has less than 3 onions
                        if len(top_soup_contents_dict['this_contents']) == 3 and len(
                                top_soup_contents_dict['other_contents']) < 3:
                            get_next_action = True

                    # placed at right counter pot
                    if (placed_obj_x, placed_obj_y) == (4, 1):
                        # hmm_actions_list.append(P2_PUT_ONION_IN_POT)
                        right_soup_contents_dict['this_contents'].append(object_held_id)
                        top_soup_contents_dict['other_contents'].append(object_held_id)

                        if p2_obj_name == 'onion':
                            both_players_strategic_action.append(2)
                            p2_major_action = 6

                        if get_next_action == True:
                            p2_time_strategic_action.append((t1, 0))
                            get_next_action = False

                        # if the right pot has 3 onions and top pot has less than 3 onions
                        if len(right_soup_contents_dict['this_contents']) == 3 and len(
                                right_soup_contents_dict['other_contents']) < 3:
                            get_next_action = True

                    object_list_tracker[object_held_id]['player_holding'] = 0
                    object_list_tracker[object_held_id]['n_actions_since_pickup'] += 1
                    object_list_tracker[object_held_id]['player_holding_list'].append(0)
                    object_list_tracker[object_held_id]['p2_n_actions_since_pickup'] += 1

                    placed_obj_x, placed_obj_y = list(p2_x_next)[0] + p2_dir_x_next, list(p2_y_next)[0] + p2_dir_y_next
                    old_obj_location = object_list_tracker[object_held_id]['location']
                    new_obj_location = (placed_obj_x, placed_obj_y)
                    object_list_tracker[object_held_id]['location'] = new_obj_location

                    object_location_tracker.pop(old_obj_location, None)
                    object_location_tracker[new_obj_location] = object_held_id

                    if (placed_obj_x, placed_obj_y) in counter_location_to_id:
                        counter_index = counter_location_to_id[(placed_obj_x, placed_obj_y)]
                        object_list_tracker[object_held_id]['counter_used'].append(counter_index)

                    object_list_tracker[object_held_id]['p2_time_completed'] = t2

                players_holding[2] = None

        # print()
        p1_actions_list.append(p1_major_action)
        p2_actions_list.append(p2_major_action)

    return object_list_tracker, ordered_delivered_tracker, p1_actions_list, p2_actions_list, p2_time_strategic_action, p2_time_strategy, hmm_actions_list


def track_p2_actions_new(p1_data, p2_data, objects_data, p1_actions,
                     p2_actions, name, time_elapsed):
    # Save player 1 and 2 actions
    p1_actions_list = []
    p2_actions_list = []

    N_steps = len(p1_data)
    a_min = 1  # the minimial value of the paramater a
    a_max = N_steps - 1  # the maximal value of the paramater a
    a_init = 1  # the value of the parameter a to be used initially, when the graph is created

    t = np.linspace(0, N_steps - 1, N_steps)

    # Define Counters
    counter_location_to_id = {
        (2, 1): 1,
        (2, 2): 2,
        (2, 3): 3,
        (1, 0): 4,
        (1, 4): 5,
        (4, 2): 6,
        (4, 3): 7
    }

    # P2 took what action when one pot had 3 onions? Fill the other or Go for a plate?
    # 0 = Fill the other
    # 1 = Go for a plate
    p2_time_strategic_action = []
    p2_time_strategy = []
    get_next_action = False

    #     p1_private_counters = [4,5]
    #     p2_private_counters = [6,7]
    #     shared_counters = [1,2,3]
    p1_private_counters = [(1, 0), (1, 4)]
    p2_private_counters = [(4, 2), (4, 3)]
    shared_counters = [(2, 1), (2, 2), (2, 3)]

    onion_dispenser_locations = [(0, 1), (0, 2)]
    dish_dispenser_locations = [(0, 3)]

    obj_count_id = 0
    next_obj_count_id = 0

    object_list_tracker = {}
    object_location_tracker = {}

    ordered_delivered_tracker = {}
    top_soup_counter_id = 0  # (3,0)
    right_soup_counter_id = 0  # (4,1)
    top_soup_contents_dict = {'this_contents': [], 'other_contents': [], 'other_state': 'empty'}  # (3,0)
    right_soup_contents_dict = {'this_contents': [], 'other_contents': [], 'other_state': 'empty'}  # (4,1)
    # states: empty, cooking, cooked, partial

    p2_carrying_soup = None
    p2_carrying_soup_pot_side = None
    p2_carrying_soup_pot_side_id = None
    absolute_order_counter = 0
    p2_time_picked_up_soup = None
    p2_time_delivered_soup = None

    players_holding = {1: None, 2: None}

    # layout = eval(old_trials[old_trials['layout_name'] == name]['layout'].to_numpy()[0])
    # layout = np.array([list(elem) for elem in layout])
    # grid_display = np.zeros((layout.shape[0], layout.shape[1], 3))

    state_actions_list = []
    state_testing_action = 0
    # Actions:
    """ (10 actions)

            1. P1 carrying onion boolean (0/1)
            2. P1 carrying dish boolean (0/1)
            3. P1 placed onion on middle boolean (0/1)
            4. P1 placed dish on middle boolean (0/1)

            5. P2 carrying onion boolean (0/1)
            6. P2 put onion in top pot (0/1)
            7. P2 put onion in right pot (0/1)
            8. P2 carrying dish boolean (0/1)
            9. P2 picked up soup from top pot (0/1)
            10. P2 picked up soup from right pot (0/1)




        """

    p1_major_action = 17  # initialize as doing nothing (stationary)
    p2_major_action = 17
    # loop over your images
    for a in range(len(t) - 1):
        state_testing_action = 0
        # for a in range(100):

        p1_x, p1_y = f_p1(t, a, p1_data)[0], f_p1(t, a, p1_data)[1]
        p1_dir_x, p1_dir_y = arrow_p1(t, a, p1_data)[2], arrow_p1(t, a, p1_data)[3]
        p1_obj_x, p1_obj_y, p1_obj_name = obj_p1(t, a, p1_data)[0], \
                                          obj_p1(t, a, p1_data)[1], \
                                          obj_p1(t, a, p1_data)[2]
        p1_act = action_p1(t, a, p1_actions)

        p2_x, p2_y = f_p2(t, a, p1_data)[0], f_p2(t, a, p1_data)[1]
        p2_dir_x, p2_dir_y = arrow_p2(t, a, p2_data)[2], arrow_p2(t, a, p2_data)[3]
        p2_obj_x, p2_obj_y, p2_obj_name = obj_p2(t, a, p2_data)[0], \
                                          obj_p2(t, a, p2_data)[1], \
                                          obj_p2(t, a, p2_data)[2]
        p2_act = action_p2(t, a, p2_actions)
        objects_list = world_obj(t, a, objects_data)
        t1 = time_elapsed[a]

        # cook_state
        # 0 = uncooked
        # 1 = cooking
        # 2 = cooked

        b = a + 1
        p1_x_next, p1_y_next = f_p1(t, b, p1_data)[0], f_p1(t, b, p1_data)[1]
        p1_dir_x_next, p1_dir_y_next = arrow_p1(t, b, p1_data)[2], arrow_p1(t, b, p1_data)[3]
        p1_obj_x_next, p1_obj_y_next, p1_obj_name_next = obj_p1(t, b, p1_data)[0], \
                                                         obj_p1(t, b, p1_data)[1], \
                                                         obj_p1(t, b, p1_data)[2]
        p1_act_next = action_p1(t, b, p1_actions)

        p2_x_next, p2_y_next = f_p2(t, b, p2_data)[0], f_p2(t, b, p2_data)[1]
        p2_dir_x_next, p2_dir_y_next = arrow_p2(t, b, p2_data)[2], arrow_p2(t, b, p2_data)[3]
        p2_obj_x_next, p2_obj_y_next, p2_obj_name_next = obj_p2(t, b, p2_data)[0], \
                                                         obj_p2(t, b, p2_data)[1], \
                                                         obj_p2(t, b, p2_data)[2]
        p2_act_next = action_p2(t, a, p2_actions)
        objects_list_next = world_obj(t, b, objects_data)
        t2 = time_elapsed[b]

        ################## PLAYER 1'S MOVEMENT ##################
        # If P1 moves or stays
        if p1_act in ['N', 'S', 'E', 'W']:
            #         print(p1_act, (p1_dir_x_next, p1_dir_y_next))

            # If P1 is carrying something and moving (dish or onion)
            if players_holding[1] is not None:
                obj_location = (p1_obj_x_next, p1_obj_y_next)
                object_held_id = players_holding[1]

                prev_location = object_list_tracker[object_held_id]['location']
                object_list_tracker[object_held_id]['n_actions_since_pickup'] += 1
                object_list_tracker[object_held_id]['location'] = obj_location
                object_list_tracker[object_held_id]['p1_n_actions_since_pickup'] += 1

                object_location_tracker.pop(prev_location, None)
                object_location_tracker[obj_location] = object_held_id

                if p1_obj_name_next == 'onion':
                    p1_major_action = 13
                    state_testing_action = 1
                elif p1_obj_name_next == 'dish':
                    p1_major_action = 14
                    state_testing_action = 2
                elif p1_obj_name_next == 'soup':
                    p1_major_action = 15
            else:
                p1_major_action = 16

        # If P1 interacted
        if p1_act == 'I':
            # If P1 picked up an object
            if p1_obj_x is None and p1_obj_x_next is not None:
                obj_location = (p1_obj_x_next, p1_obj_y_next)
                object_location_tracker[obj_location] = obj_count_id

                if obj_count_id not in object_list_tracker:
                    object_list_tracker[obj_count_id] = {}
                object_list_tracker[obj_count_id]['name'] = p1_obj_name_next
                object_list_tracker[obj_count_id]['player_holding'] = 1
                object_list_tracker[obj_count_id]['id'] = obj_count_id
                object_list_tracker[obj_count_id]['n_actions_since_pickup'] = 0
                object_list_tracker[obj_count_id]['location'] = obj_location
                object_list_tracker[obj_count_id]['on_screen'] = True
                object_list_tracker[obj_count_id]['player_holding_list'] = [1]
                object_list_tracker[obj_count_id]['p1_n_actions_since_pickup'] = 1
                object_list_tracker[obj_count_id]['p2_n_actions_since_pickup'] = 0
                object_list_tracker[obj_count_id]['counter_used'] = []
                object_list_tracker[obj_count_id]['p1_time_started'] = t1
                object_list_tracker[obj_count_id]['p1_time_completed'] = None
                object_list_tracker[obj_count_id]['p2_time_started'] = None
                object_list_tracker[obj_count_id]['p2_time_completed'] = None

                players_holding[1] = obj_count_id

                obj_count_id += 1

                picked_up_from_x, picked_up_from_y = list(p1_x_next)[0] + p1_dir_x_next, list(p1_y_next)[
                    0] + p1_dir_y_next
                picked_up_loc = (picked_up_from_x, picked_up_from_y)

                if p1_obj_name_next == 'onion':
                    state_testing_action = 1
                    if picked_up_loc in onion_dispenser_locations:
                        p1_major_action = 1
                    if picked_up_loc in p1_private_counters:
                        p1_major_action = 20
                    if picked_up_loc in shared_counters:
                        p1_major_action = 18


                elif p1_obj_name_next == 'dish':
                    state_testing_action = 2
                    if picked_up_loc in dish_dispenser_locations:
                        p1_major_action = 2
                    if picked_up_loc in p1_private_counters:
                        p1_major_action = 21
                    if picked_up_loc in shared_counters:
                        p1_major_action = 19

                elif p1_obj_name_next == 'soup':
                    if picked_up_loc in p1_private_counters:
                        p1_major_action = 22
                    if picked_up_loc in shared_counters:
                        p1_major_action = 23

            # If P1 put down an object
            if p1_obj_x is not None and p1_obj_x_next is None:
                object_held_id = players_holding[1]

                object_list_tracker[object_held_id]['player_holding'] = 0
                object_list_tracker[object_held_id]['n_actions_since_pickup'] += 1
                object_list_tracker[object_held_id]['player_holding_list'].append(0)
                object_list_tracker[object_held_id]['p1_n_actions_since_pickup'] += 1

                placed_obj_x, placed_obj_y = list(p1_x_next)[0] + p1_dir_x_next, list(p1_y_next)[0] + p1_dir_y_next
                old_obj_location = object_list_tracker[object_held_id]['location']
                new_obj_location = (placed_obj_x, placed_obj_y)
                object_list_tracker[object_held_id]['location'] = new_obj_location

                object_location_tracker.pop(old_obj_location, None)
                object_location_tracker[new_obj_location] = object_held_id

                counter_index = counter_location_to_id[(placed_obj_x, placed_obj_y)]
                object_list_tracker[object_held_id]['counter_used'].append(counter_index)

                object_list_tracker[object_held_id]['p1_time_completed'] = t2

                players_holding[1] = None

                put_down_loc = new_obj_location
                if p1_obj_name == 'onion':
                    state_testing_action = 3
                    if put_down_loc in p1_private_counters:
                        p1_major_action = 10
                    if put_down_loc in shared_counters:
                        p1_major_action = 3


                elif p1_obj_name == 'dish':
                    state_testing_action = 4
                    if put_down_loc in p1_private_counters:
                        p1_major_action = 11
                    if put_down_loc in shared_counters:
                        p1_major_action = 4

                elif p1_obj_name == 'soup':
                    if put_down_loc in p1_private_counters:
                        p1_major_action = 12
                    if put_down_loc in shared_counters:
                        p1_major_action = 24

        ################## PLAYER 2'S MOVEMENT ##################
        # If P2 moves or stays
        if p2_act in ['N', 'S', 'E', 'W']:

            # If P2 is moving with an object: onion, dish, or soup
            if players_holding[2] is not None:
                obj_location = (p2_obj_x_next, p2_obj_y_next)
                object_held_id = players_holding[2]

                prev_location = object_list_tracker[object_held_id]['location']
                object_list_tracker[object_held_id]['n_actions_since_pickup'] += 1
                object_list_tracker[object_held_id]['location'] = obj_location
                object_list_tracker[object_held_id]['p2_n_actions_since_pickup'] += 1

                object_location_tracker.pop(prev_location, None)
                object_location_tracker[obj_location] = object_held_id

                if p2_obj_name_next == 'onion':
                    p2_major_action = 13
                    state_testing_action = 5
                elif p2_obj_name_next == 'dish':
                    p2_major_action = 14
                    state_testing_action = 8
                elif p2_obj_name_next == 'soup':
                    p2_major_action = 15
            else:
                p2_major_action = 16

        # If P2 interacted
        if p2_act == 'I':
            placed_x, placed_y = list(p2_x_next)[0] + p2_dir_x_next, list(p2_y_next)[0] + p2_dir_y_next
            # If P2 picked up soup from the pot. P2 would be carrying a dish
            if p2_obj_name_next == 'soup':

                # P2 was already carrying this soup
                #                 if p2_obj_name == 'soup':
                #                     print('WHAT IS GOING ON', (p2_data[a], p2_actions[a]))
                #                     print('WHAT IS GOING ON NEXT', (p2_data[b], p2_actions[b]))
                #                     print('picked up from', (placed_x, placed_y))
                #                     print()

                # P2 filled up a dish with the soup
                if p2_obj_name == 'dish':
                    objects_status = objects_data[a]
                    objects_status_next = objects_data[b]
                    top_pot_status = 'empty'
                    right_pot_status = 'empty'
                    for obj_status in objects_status:
                        if obj_status['name'] == 'soup' and obj_status['position'] == [3, 0]:
                            if obj_status['is_cooking'] == True:
                                top_pot_status = 'cooking'
                            if obj_status['is_ready'] == True:
                                top_pot_status = 'ready'
                            if obj_status['is_idle'] == True:
                                top_pot_status = 'idle'
                        if obj_status['name'] == 'soup' and obj_status['position'] == [4, 1]:
                            if obj_status['is_cooking'] == True:
                                right_pot_status = 'cooking'
                            if obj_status['is_ready'] == True:
                                right_pot_status = 'ready'
                            if obj_status['is_idle'] == True:
                                right_pot_status = 'idle'

                    top_soup_contents_dict['other_state'] = right_pot_status
                    right_soup_contents_dict['other_state'] = top_pot_status

                    p2_time_picked_up_soup = t1

                    # picked up from top counter
                    if (placed_x, placed_y) == (3, 0):
                        state_testing_action = 9
                        p2_carrying_soup = copy.deepcopy(top_soup_contents_dict)
                        p2_carrying_soup_pot_side = 'top'
                        p2_carrying_soup_pot_side_id = top_soup_counter_id

                        top_soup_contents_dict['this_contents'] = []  # (3,0)
                        right_soup_contents_dict['other_contents'] = []
                        right_soup_contents_dict['other_state'] = 'empty'
                        top_soup_counter_id += 1

                        p2_major_action = 7

                    #                     objects_data_next [{'name': 'soup', 'position': [3, 0], '_ingredients': [{'name': 'onion', 'position': [3, 0]}], 'cooking_tick': -1, 'is_cooking': False, 'is_ready': False, 'is_idle': True, 'cook_time': -1, '_cooking_tick': -1}]

                    # picked up from right counter
                    if (placed_x, placed_y) == (4, 1):
                        state_testing_action = 10
                        p2_carrying_soup = copy.deepcopy(right_soup_contents_dict)
                        p2_carrying_soup_pot_side = 'right'
                        p2_carrying_soup_pot_side_id = right_soup_counter_id

                        right_soup_contents_dict['this_contents'] = []  # (4,1)
                        top_soup_contents_dict['other_contents'] = []
                        top_soup_contents_dict['other_state'] = 'empty'
                        right_soup_counter_id += 1

                        p2_major_action = 8

            # If P2 picked up an object
            if p2_obj_x is None and p2_obj_x_next is not None:
                placed_x, placed_y = list(p2_x_next)[0] + p2_dir_x_next, list(p2_y_next)[0] + p2_dir_y_next

                if (placed_x, placed_y) in object_location_tracker:
                    obj_picked_id = object_location_tracker[(placed_x, placed_y)]
                else:
                    print('!!! problem p2 pickup not found')
                    nearest_key = min(list(object_location_tracker.keys()),
                                      key=lambda c: (c[0] - placed_x) ** 2 + (c[1] - placed_y) ** 2)
                    obj_picked_id = object_location_tracker[nearest_key]

                new_obj_location = (p2_obj_x_next, p2_obj_y_next)

                object_list_tracker[obj_picked_id]['player_holding'] = 2
                object_list_tracker[obj_picked_id]['n_actions_since_pickup'] += 1
                object_list_tracker[obj_picked_id]['location'] = new_obj_location
                object_list_tracker[obj_picked_id]['on_screen'] = True
                object_list_tracker[obj_picked_id]['player_holding_list'].append(2)
                object_list_tracker[obj_picked_id]['p2_n_actions_since_pickup'] += 1
                object_list_tracker[obj_picked_id]['p2_time_started'] = t1

                players_holding[2] = obj_picked_id

                object_location_tracker.pop((placed_x, placed_y), None)
                object_location_tracker[new_obj_location] = obj_picked_id

                # new
                picked_up_loc = (placed_x, placed_y)

                if p2_obj_name_next == 'onion':
                    state_testing_action = 5
                    if picked_up_loc in p2_private_counters:
                        p2_major_action = 20
                    if picked_up_loc in shared_counters:
                        p2_major_action = 18


                elif p2_obj_name_next == 'dish':
                    state_testing_action = 8
                    if picked_up_loc in p2_private_counters:
                        p2_major_action = 21
                    if picked_up_loc in shared_counters:
                        p2_major_action = 19

                elif p2_obj_name_next == 'soup':
                    if picked_up_loc in p2_private_counters:
                        p2_major_action = 22
                    if picked_up_loc in shared_counters:
                        p2_major_action = 23

        # If P2 put down an object
        if p2_obj_x is not None and p2_obj_x_next is None:
            object_held_id = players_holding[2]
            placed_obj_x, placed_obj_y = list(p2_x_next)[0] + p2_dir_x_next, list(p2_y_next)[0] + p2_dir_y_next

            # If P2 delivered a soup
            if p2_obj_name == 'soup' and (placed_obj_x, placed_obj_y) == (3, 4):
                p2_time_delivered_soup = t2
                ordered_delivered_tracker[absolute_order_counter] = {}
                ordered_delivered_tracker[absolute_order_counter]['details'] = p2_carrying_soup
                ordered_delivered_tracker[absolute_order_counter]['pot_side'] = p2_carrying_soup_pot_side
                ordered_delivered_tracker[absolute_order_counter]['pot_side_id'] = p2_carrying_soup_pot_side_id
                ordered_delivered_tracker[absolute_order_counter]['time_picked_up'] = p2_time_picked_up_soup
                ordered_delivered_tracker[absolute_order_counter]['time_delivered'] = p2_time_delivered_soup

                absolute_order_counter += 1
                p2_major_action = 9

                p2_time_strategy.append(t1)

                if get_next_action == True:
                    p2_time_strategic_action.append((t1, 1))
                    get_next_action = False


            else:

                # placed at top counter pot
                if (placed_obj_x, placed_obj_y) == (3, 0):
                    state_testing_action = 6
                    top_soup_contents_dict['this_contents'].append(object_held_id)
                    right_soup_contents_dict['other_contents'].append(object_held_id)

                    if p2_obj_name == 'onion':
                        p2_major_action = 5

                    if get_next_action == True:
                        p2_time_strategic_action.append((t1, 0))
                        get_next_action = False

                    # if the top pot has 3 onions and right pot has less than 3 onions
                    if len(top_soup_contents_dict['this_contents']) == 3 and len(
                            top_soup_contents_dict['other_contents']) < 3:
                        get_next_action = True

                # placed at right counter pot
                if (placed_obj_x, placed_obj_y) == (4, 1):
                    state_testing_action = 7
                    right_soup_contents_dict['this_contents'].append(object_held_id)
                    top_soup_contents_dict['other_contents'].append(object_held_id)

                    if p2_obj_name == 'onion':
                        p2_major_action = 6

                    if get_next_action == True:
                        p2_time_strategic_action.append((t1, 0))
                        get_next_action = False

                    # if the right pot has 3 onions and top pot has less than 3 onions
                    if len(right_soup_contents_dict['this_contents']) == 3 and len(
                            right_soup_contents_dict['other_contents']) < 3:
                        get_next_action = True

                object_list_tracker[object_held_id]['player_holding'] = 0
                object_list_tracker[object_held_id]['n_actions_since_pickup'] += 1
                object_list_tracker[object_held_id]['player_holding_list'].append(0)
                object_list_tracker[object_held_id]['p2_n_actions_since_pickup'] += 1

                placed_obj_x, placed_obj_y = list(p2_x_next)[0] + p2_dir_x_next, list(p2_y_next)[0] + p2_dir_y_next
                old_obj_location = object_list_tracker[object_held_id]['location']
                new_obj_location = (placed_obj_x, placed_obj_y)
                object_list_tracker[object_held_id]['location'] = new_obj_location

                object_location_tracker.pop(old_obj_location, None)
                object_location_tracker[new_obj_location] = object_held_id

                if (placed_obj_x, placed_obj_y) in counter_location_to_id:
                    counter_index = counter_location_to_id[(placed_obj_x, placed_obj_y)]
                    object_list_tracker[object_held_id]['counter_used'].append(counter_index)

                object_list_tracker[object_held_id]['p2_time_completed'] = t2

            players_holding[2] = None

        p1_actions_list.append(p1_major_action)
        p2_actions_list.append(p2_major_action)
        if state_testing_action != 0:
            state_actions_list.append(state_testing_action)

    return object_list_tracker, ordered_delivered_tracker, p1_actions_list, p2_actions_list, p2_time_strategic_action, p2_time_strategy, state_actions_list



def featurize_data_for_chunked_naive_hmm(team_id_to_state_action_sequences, window=20, ss=5):
    team_chunked_actions_data = {}

    name = 'random0'
    title = 'Forced Coord'

    trial_data = {}
    team_numbers = []
    for j in team_id_to_state_action_sequences:
        # for j in [5]:
        trial_id = j
        team_numbers.append(trial_id)
        team_specific_data = team_id_to_state_action_sequences[j]
        # print('trial_id', trial_id)
        # trial_df = old_trials[old_trials['trial_id'] == trial_id]
        score = team_specific_data['score']
        state_data = team_specific_data['states'][0]
        joint_actions = team_specific_data['actions'][0]
        time_elapsed = team_specific_data['time_elapsed']
        # time_elapsed = trial_df['time_elapsed'].to_numpy()
        # print('\n\nlen state_data', len(state_data))
        # print('state_data', state_data)

        p1_data = []
        p2_data = []
        p1_actions = []
        p2_actions = []
        state_data_eval = []
        objects_data = []
        for i in range(1, len(state_data)):
            prev_state_x = state_data[i-1].to_dict()
            state_x = state_data[i].to_dict()
            joint_actions_i = joint_actions[i]
            p1_index = 1
            p2_index = 0

            p1_data.append(state_x['players'][p1_index])
            p2_data.append(state_x['players'][p2_index])
            state_data_eval.append(state_x)
            objects_data.append(state_x['objects'])

            p1_actions.append(joint_actions_i[p1_index])
            p2_actions.append(joint_actions_i[p2_index])

        object_list_tracker, ordered_delivered_tracker, p1_actions_list, p2_actions_list, p2_time_strategic_action, p2_time_strategy, hmm_actions = track_p2_actions(p1_data, p2_data, objects_data, p1_actions,
            p2_actions, name, time_elapsed)

        # object_list_tracker, ordered_delivered_tracker, p1_actions_list, p2_actions_list, p2_time_strategic_action, p2_time_strategy, state_actions_list = track_p2_actions(p1_data, p2_data, objects_data, p1_actions,
        #     p2_actions, name, time_elapsed)

        team_chunked_actions_data[trial_id] = {}
        team_chunked_actions_data[trial_id]['p1_actions'] = p1_actions_list
        team_chunked_actions_data[trial_id]['p2_actions'] = p2_actions_list
        team_chunked_actions_data[trial_id]['p2_major_actions'] = p2_time_strategic_action
        team_chunked_actions_data[trial_id]['p2_delivery_times'] = p2_time_strategy
        team_chunked_actions_data[trial_id]['state_actions_list'] = hmm_actions
    #     other_pot_contains_num_onions, other_pot_states, steps_p1_took_total_order, \
    #         steps_p2_took_total_order, order_completion_times = pull_features_from_output(object_list_tracker, ordered_delivered_tracker)

    #     plot_results(other_pot_contains_num_onions, other_pot_states, steps_p1_took_total_order, steps_p2_took_total_order, order_completion_times)
    #     print()
    # team_idx = 9
    # print("team_chunked_actions_data", team_chunked_actions_data)

    X_data = []
    team_numbers = []

    X_data_chunked = []
    team_numbers_chunked = []


    for team_idx in team_chunked_actions_data:

        state_actions_list = team_chunked_actions_data[team_idx]['state_actions_list']
        # p2_delivery_times = team_chunked_actions_data[team_idx]['p2_delivery_times']

        #     print(len(p2_major_actions))
        # add = []
        # for j in range(5):
        #     add.append(p2_major_actions[j][1])

        # print("state_actions_list", np.unique(state_actions_list))

        # add = []
        # add = [p2_major_actions[c][1] for c in range(len(p2_major_actions))]
        # add = [state_actions_list[c] for c in range(len(state_actions_list))]
        add =  [state_actions_list[c] for c in range(50)]
        # add = []
        # for j in range(len(state_actions_list)):
        #     add.append(state_actions_list[j] - 1)
        X_data.append(np.array(add))

        # X_data.append(add)
        # X_data.append(np.array(add))
        team_numbers.append(team_idx)

        for j in range(0, len(state_actions_list)-window, ss):
            add = [state_actions_list[c] for c in range(j, j+window)]
            X_data_chunked.append(np.array(add))
            team_numbers_chunked.append(team_idx)

    # X_data = np.array(X_data)

    return X_data, team_numbers, X_data_chunked, team_numbers_chunked

def run_naive_hmm_on_p2(team_id_to_state_action_sequences, n_states, window=4, ss=2):
    # print('team_id_to_state_action_sequences', team_id_to_state_action_sequences)
    # X = observation_data
    # Y = hidden_state_data
    X, team_numbers, X_data_chunked, team_numbers_chunked = featurize_data_for_chunked_naive_hmm(team_id_to_state_action_sequences, window=window, ss=ss)
    # print("X = ", X)
    # X = X_data_chunked
    # team_numbers = team_numbers_chunked
    print(np.unique(X))

    N_iters = 100

    test_unsuper_hmm = unsupervised_HMM(X, n_states, N_iters)

    # print('emission', test_unsuper_hmm.generate_emission(10))
    hidden_seqs = []
    team_num_to_seq_probs = {}
    for j in range(len(X)):
        viterbi_output, all_sequences_and_probs = test_unsuper_hmm.viterbi_all_probs(X[j])
        team_num_to_seq_probs[team_numbers[j]] = all_sequences_and_probs
        hidden_seqs.append([int(x) for x in viterbi_output])
        print('viterbi: hidden seq: Team ' + str(team_numbers[j]) + ": ", viterbi_output)

    return test_unsuper_hmm, hidden_seqs, team_numbers, team_num_to_seq_probs

def pad_w_mode(hidden_seqs):
    max_len = max(len(elem) for elem in hidden_seqs)
    X = []
    for i in range(len(hidden_seqs)):
        team_hs = hidden_seqs[i]
        team_hs_pad = []
        team_hs_mode = max(set(team_hs), key=team_hs.count)
        for j in range(max_len):
            if j < len(team_hs):
                team_hs_pad.append(team_hs[j])
            else:
                team_hs_pad.append(team_hs_mode)
        X.append(team_hs_pad)
    # print('X', X)
    return np.array(X)



def cluster_hidden_states(hidden_seqs, n_clusters=2):
    # X = pad_w_mode(hidden_seqs)
    X = hidden_seqs
    # print('X=', X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    ss = sklearn.metrics.silhouette_score(X, cluster_labels)
    # ss = sklearn.metrics.calinski_harabasz_score(X, cluster_labels)
    return cluster_labels, cluster_centers, ss


def cluster_hidden_states_new(hidden_seqs, n_clusters=2):
    X = np.array(hidden_seqs)
    # print('X=', X)
    ## METHOD 1: Euclidean
    # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    # cluster_labels = kmeans.labels_
    # cluster_centers = kmeans.cluster_centers_

    ## METHOD 2: Leven, Edit Distance
    cluster_labels, cluster_centers = perform_clustering_on_int_strings(X, n_clusters)
    print('cluster_labels', cluster_labels)
    print('cluster_centers', cluster_centers)

    ss = sklearn.metrics.silhouette_score(X, cluster_labels)
    # ss = sklearn.metrics.calinski_harabasz_score(X, cluster_labels)
    return cluster_labels, cluster_centers, ss


def create_fake_dataset(window=4, ss=2, num_samples=10, seq_len=5):
    X, team_numbers = featurize_data_for_chunked_naive_hmm(window=window, ss=ss)

    simulated_X = []
    simulated_team_labels = []
    simulated_strategy_labels = []

    for n in range(num_samples):
        rand_indices = np.random.choice(range(len(X)), size=seq_len)
        strat_1_index = team_numbers.index(79)
        alternating_rand_indices = []
        for value in rand_indices:
            alternating_rand_indices.append(value)
            alternating_rand_indices.append(strat_1_index)

        print(alternating_rand_indices)
        fake_seq = []
        fake_seq_team_numbers = []
        for idx in rand_indices:
            fake_seq.extend(list(X[idx]))

            fake_seq_team_numbers.extend([team_numbers[idx] for t in range(len(X[idx]))])

        for j in range(0, len(fake_seq) - window, ss):
            add = [fake_seq[c] for c in range(j, j + window)]
            simulated_X.append(np.array(add))
            team_idx = fake_seq_team_numbers[j+window]
            if int(team_idx) == 79 or int(team_idx) == 114:
                simulated_strategy_labels.append(1)
            else:
                simulated_strategy_labels.append(0)
            simulated_team_labels.append(team_idx)

    return simulated_X, simulated_team_labels, simulated_strategy_labels


def get_predicted_team_strat(team_num_to_seq_probs, simulated_X, simulated_team_labels, simulated_strategy_labels):
    predicted_team_labels = []
    predicted_strat_labels = []
    for j in range(len(simulated_X)):
        find_X = simulated_X[j]
        team_list = []
        probs_list = []

        probability_of_strat_0 = 0
        probability_of_strat_1 = 0
        for team_number in team_num_to_seq_probs:
            seq_probs = team_num_to_seq_probs[team_number]
            hidden_for_find_X = test_unsuper_hmm.viterbi(find_X)
            print('hidden_for_find_X', hidden_for_find_X)
            if hidden_for_find_X not in seq_probs:
                print("problem", seq_probs)
                continue
            sequence_prob_for_team_i = seq_probs[hidden_for_find_X]
            team_list.append(team_number)
            probs_list.append(sequence_prob_for_team_i)
            if team_number == 79 or team_number == 114:
                probability_of_strat_1 += sequence_prob_for_team_i
            else:
                probability_of_strat_0 += sequence_prob_for_team_i

        if len(probs_list) == 0:
            predicted_team_labels.append(predicted_team_labels[-1])
            predicted_strat_labels.append(predicted_strat_labels[-1])
        else:
            max_idx = np.argmax(probs_list)
            max_team = team_list[max_idx]
            max_prob = probs_list[max_idx]
            # if int(max_team) == 79 or int(max_team) == 114:
            #     max_strat = 1
            # else:
            #     max_strat = 0
            if probability_of_strat_1 > probability_of_strat_0:
                max_strat = 1
            else:
                max_strat = 0

            predicted_team_labels.append(max_team)
            predicted_strat_labels.append(max_strat)

    team_acc = 0
    strat_acc = 0
    for i in range(len(predicted_team_labels)):
        if int(predicted_team_labels[i]) == int(simulated_team_labels[i]):
            team_acc += 1
        if int(predicted_strat_labels[i]) == int(simulated_strategy_labels[i]):
            strat_acc += 1


    team_acc /= len(predicted_team_labels)
    strat_acc /= len(predicted_team_labels)
    return team_acc, strat_acc, predicted_team_labels, predicted_strat_labels


def generate_synthetic_fixed(N_teams=6):
    team_id_to_state_action_sequences = {}
    for i in range(N_teams):
        fixed_results, avg_fixed_results = run_one_game_fixed_agents('DP', 'DP')
        action_sequence = fixed_results['ep_actions']
        state_sequence = fixed_results['ep_observations']
        team_id_to_state_action_sequences[i] = {}
        team_id_to_state_action_sequences[i]['states'] = state_sequence
        team_id_to_state_action_sequences[i]['actions'] = action_sequence
        team_id_to_state_action_sequences[i]['true_class'] = 0
        team_id_to_state_action_sequences[i]['score'] = avg_fixed_results
        team_id_to_state_action_sequences[i]['time_elapsed'] = list(range(400))

    for i in range(N_teams):
        fixed_results, avg_fixed_results = run_one_game_fixed_agents('SP', 'SP')
        action_sequence = fixed_results['ep_actions']
        state_sequence = fixed_results['ep_observations']
        team_id_to_state_action_sequences[N_teams + i] = {}
        team_id_to_state_action_sequences[N_teams + i]['states'] = state_sequence
        team_id_to_state_action_sequences[N_teams + i]['actions'] = action_sequence
        team_id_to_state_action_sequences[N_teams + i]['true_class'] = 1
        team_id_to_state_action_sequences[N_teams + i]['score'] = avg_fixed_results
        team_id_to_state_action_sequences[N_teams + i]['time_elapsed'] = list(range(400))

    return team_id_to_state_action_sequences



def run_HMM_on_fixed(team_id_to_state_action_sequences):
    # num_states_list = [2,3,4]
    # num_clusters_list = [2, 3,4]
    num_states_list = [2, 3, 4, 5, 6, 7, 8, 9]
    num_clusters_list = [2]

    arr = np.zeros((10, 10))

    for n_states in num_states_list:
        test_unsuper_hmm, hidden_seqs, team_numbers, team_num_to_seq_probs = run_naive_hmm_on_p2(team_id_to_state_action_sequences, n_states=n_states, window=50, ss=20)

        # print('hidden_seqs', hidden_seqs)

        for n_clusters in num_clusters_list:



            cluster_labels, cluster_centers, ss = cluster_hidden_states(hidden_seqs, n_clusters=n_clusters)
            print(f'\nN={n_states}, K={n_clusters}: cluster_labels = {cluster_labels}, teams = {team_numbers}')
            print(f'N={n_states}, K={n_clusters}, sil. score = ', ss)
            arr[n_states, n_clusters] = ss


    # ax = plt.figure()
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    plt.title("Forced Coordination")
    plt.ylabel("Number of Hidden States")
    plt.xlabel("Number of Clusters")
    plt.xlim(num_clusters_list[0]-0.5, num_clusters_list[-1]+0.5)
    plt.ylim(num_states_list[0]-0.5, num_states_list[-1]+0.5)
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig('fc_ss_score_fixed_1.png')
    plt.close()








if __name__ == "__main__":
    team_id_to_state_action_sequences = generate_synthetic_fixed(N_teams=6)
    run_HMM_on_fixed(team_id_to_state_action_sequences)
    # print('action_sequence', action_sequence)







