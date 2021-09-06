import numpy as np
from dependencies import *
from hmm import supervised_HMM, unsupervised_HMM, HiddenMarkovModel
from extract_features import *
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import sklearn

def track_p2_actions(old_trials, p1_data, p2_data, objects_data, p1_actions,
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

    layout = eval(old_trials[old_trials['layout_name'] == name]['layout'].to_numpy()[0])
    layout = np.array([list(elem) for elem in layout])
    grid_display = np.zeros((layout.shape[0], layout.shape[1], 3))

    p1_major_action = 17  # initialize as doing nothing (stationary)
    p2_major_action = 17
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
                    both_players_strategic_action.append(0)
                    if put_down_loc in p1_private_counters:
                        p1_major_action = 10
                    if put_down_loc in shared_counters:
                        p1_major_action = 3


                elif p1_obj_name == 'dish':
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

                both_players_strategic_action.append(3)

                if get_next_action == True:
                    p2_time_strategic_action.append((t1, 1))
                    get_next_action = False


            else:

                # placed at top counter pot
                if (placed_obj_x, placed_obj_y) == (3, 0):

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

        p1_actions_list.append(p1_major_action)
        p2_actions_list.append(p2_major_action)

    return object_list_tracker, ordered_delivered_tracker, p1_actions_list, p2_actions_list, p2_time_strategic_action, p2_time_strategy




def featurize_data_for_naive_hmm():
    team_chunked_actions_data = {}

    name = 'random0'
    title = 'Forced Coord'

    # name = 'random0'
    # title = 'Forced Coordination'
    old_trials = import_2019_data()
    layout_trials = old_trials[old_trials['layout_name'] == name]['trial_id'].unique()
    # name = 'random0'
    # title = 'Forced Coordination'
    trial_data = {}
    team_numbers = []
    for j in range(len(layout_trials)):
        # for j in [5]:
        trial_id = layout_trials[j]
        team_numbers.append(trial_id)
        print('trial_id', trial_id)
        trial_df = old_trials[old_trials['trial_id'] == trial_id]
        score = old_trials[old_trials['trial_id'] == trial_id]['score'].to_numpy()[-1]
        state_data = trial_df['state'].to_numpy()
        joint_actions = trial_df['joint_action'].to_numpy()
        time_elapsed = trial_df['time_elapsed'].to_numpy()

        p1_data = []
        p2_data = []
        p1_actions = []
        p2_actions = []
        state_data_eval = []
        objects_data = []
        for i in range(1, len(state_data)):
            prev_state_x = json_eval(state_data[i - 1])
            state_x = json_eval(state_data[i])
            joint_actions_i = literal_eval(joint_actions[i])
            p1_index = 1
            p2_index = 0

            p1_data.append(state_x['players'][p1_index])
            p2_data.append(state_x['players'][p2_index])
            state_data_eval.append(state_x)
            objects_data.append(state_x['objects'])

            p1_actions.append(joint_actions_i[p1_index])
            p2_actions.append(joint_actions_i[p2_index])

        object_list_tracker, ordered_delivered_tracker, p1_actions_list, p2_actions_list, p2_time_strategic_action, p2_time_strategy = track_p2_actions(old_trials,
            p1_data, p2_data, objects_data, p1_actions,
            p2_actions, name, time_elapsed)
        team_chunked_actions_data[trial_id] = {}
        team_chunked_actions_data[trial_id]['p1_actions'] = p1_actions_list
        team_chunked_actions_data[trial_id]['p2_actions'] = p2_actions_list
        team_chunked_actions_data[trial_id]['p2_major_actions'] = p2_time_strategic_action
        team_chunked_actions_data[trial_id]['p2_delivery_times'] = p2_time_strategy
    #     other_pot_contains_num_onions, other_pot_states, steps_p1_took_total_order, \
    #         steps_p2_took_total_order, order_completion_times = pull_features_from_output(object_list_tracker, ordered_delivered_tracker)

    #     plot_results(other_pot_contains_num_onions, other_pot_states, steps_p1_took_total_order, steps_p2_took_total_order, order_completion_times)
    #     print()
    # team_idx = 9

    X_data = []
    for team_idx in team_chunked_actions_data:

        p2_major_actions = team_chunked_actions_data[team_idx]['p2_major_actions']
        p2_delivery_times = team_chunked_actions_data[team_idx]['p2_delivery_times']

        #     print(len(p2_major_actions))
        # add = []
        # for j in range(5):
        #     add.append(p2_major_actions[j][1])
        add = [p2_major_actions[j][1] for j in range(len(p2_major_actions))]
        X_data.append(np.array(add))

    # X_data = np.array(X_data)

    return X_data, team_numbers


def featurize_data_for_chunked_naive_hmm(window=4, ss=2):
    team_chunked_actions_data = {}

    name = 'random0'
    title = 'Forced Coord'

    # name = 'random0'
    # title = 'Forced Coordination'
    old_trials = import_2019_data()
    layout_trials = old_trials[old_trials['layout_name'] == name]['trial_id'].unique()
    # name = 'random0'
    # title = 'Forced Coordination'
    trial_data = {}
    team_numbers = []
    for j in range(len(layout_trials)):
        # for j in [5]:
        trial_id = layout_trials[j]
        team_numbers.append(trial_id)
        # print('trial_id', trial_id)
        trial_df = old_trials[old_trials['trial_id'] == trial_id]
        score = old_trials[old_trials['trial_id'] == trial_id]['score'].to_numpy()[-1]
        state_data = trial_df['state'].to_numpy()
        joint_actions = trial_df['joint_action'].to_numpy()
        time_elapsed = trial_df['time_elapsed'].to_numpy()

        p1_data = []
        p2_data = []
        p1_actions = []
        p2_actions = []
        state_data_eval = []
        objects_data = []
        for i in range(1, len(state_data)):
            prev_state_x = json_eval(state_data[i - 1])
            state_x = json_eval(state_data[i])
            joint_actions_i = literal_eval(joint_actions[i])
            p1_index = 1
            p2_index = 0

            p1_data.append(state_x['players'][p1_index])
            p2_data.append(state_x['players'][p2_index])
            state_data_eval.append(state_x)
            objects_data.append(state_x['objects'])

            p1_actions.append(joint_actions_i[p1_index])
            p2_actions.append(joint_actions_i[p2_index])

        object_list_tracker, ordered_delivered_tracker, p1_actions_list, p2_actions_list, p2_time_strategic_action, p2_time_strategy = track_p2_actions(old_trials,
            p1_data, p2_data, objects_data, p1_actions,
            p2_actions, name, time_elapsed)
        team_chunked_actions_data[trial_id] = {}
        team_chunked_actions_data[trial_id]['p1_actions'] = p1_actions_list
        team_chunked_actions_data[trial_id]['p2_actions'] = p2_actions_list
        team_chunked_actions_data[trial_id]['p2_major_actions'] = p2_time_strategic_action
        team_chunked_actions_data[trial_id]['p2_delivery_times'] = p2_time_strategy
    #     other_pot_contains_num_onions, other_pot_states, steps_p1_took_total_order, \
    #         steps_p2_took_total_order, order_completion_times = pull_features_from_output(object_list_tracker, ordered_delivered_tracker)

    #     plot_results(other_pot_contains_num_onions, other_pot_states, steps_p1_took_total_order, steps_p2_took_total_order, order_completion_times)
    #     print()
    # team_idx = 9

    X_data = []
    team_numbers = []

    X_data_chunked = []
    team_numbers_chunked = []


    for team_idx in team_chunked_actions_data:

        p2_major_actions = team_chunked_actions_data[team_idx]['p2_major_actions']
        p2_delivery_times = team_chunked_actions_data[team_idx]['p2_delivery_times']

        #     print(len(p2_major_actions))
        # add = []
        # for j in range(5):
        #     add.append(p2_major_actions[j][1])

        # add = []
        add = [p2_major_actions[c][1] for c in range(len(p2_major_actions))]
        X_data.append(np.array(add))
        team_numbers.append(team_idx)

        for j in range(0, len(p2_major_actions)-window, ss):
            add = [p2_major_actions[c][1] for c in range(j, j+window)]
            X_data_chunked.append(np.array(add))
            team_numbers_chunked.append(team_idx)

    # X_data = np.array(X_data)

    return X_data, team_numbers, X_data_chunked, team_numbers_chunked

def run_naive_hmm_on_p2(n_states, window=4, ss=2):

    # X = observation_data
    # Y = hidden_state_data
    X, team_numbers, X_data_chunked, team_numbers_chunked = featurize_data_for_chunked_naive_hmm(window=window, ss=ss)


    N_iters = 100

    test_unsuper_hmm = unsupervised_HMM(X, n_states, N_iters)

    # print('emission', test_unsuper_hmm.generate_emission(10))
    hidden_seqs = []
    team_num_to_seq_probs = {}
    for j in range(len(X)):
        viterbi_output, all_sequences_and_probs = test_unsuper_hmm.viterbi_all_probs(X[j])
        team_num_to_seq_probs[team_numbers[j]] = all_sequences_and_probs
        hidden_seqs.append([int(x) for x in viterbi_output])
        # print('viterbi: hidden seq: Team ' + str(team_numbers[j]) + ": ", viterbi_output)

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
    X = pad_w_mode(hidden_seqs)
    # print('X=', X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
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


def online_prediction_testing():
    test_unsuper_hmm, hidden_seqs, team_numbers, team_num_to_seq_probs = run_naive_hmm_on_p2(n_states=3, window=7, ss=3)

    # Try N=2 Clusters
    cluster_labels, cluster_centers = cluster_hidden_states(hidden_seqs, n_clusters=2)
    print(f'\nN=2: cluster_labels = {[(team_numbers[i], cluster_labels[i]) for i in range(len(cluster_labels))]}, teams = {team_numbers}, cluster_centers = {cluster_centers}')

    # Try N=3 Clusters
    # cluster_labels, cluster_centers = cluster_hidden_states(hidden_seqs, n_clusters=3)
    # print(f'\nN=3: cluster_labels = {[(team_numbers[i], cluster_labels[i]) for i in range(len(cluster_labels))]}, cluster_centers = {cluster_centers}')

    plot_window_size = [3,4,5,6,7]
    strat_accuracies = []
    strat_stds = []
    n_reps = 1
    strat_acc_means = []
    strat_acc_stds = []

    strat_recall_means = []
    strat_recall_stds = []

    strat_prec_means = []
    strat_prec_stds = []


    for win_size in plot_window_size:

        strat_acc_list = []
        strat_recall_list = []
        strat_prec_list = []
        for rep in range(n_reps):
            simulated_X, simulated_team_labels, simulated_strategy_labels = create_fake_dataset(window=win_size, ss=3, num_samples=10, seq_len=6)

            team_acc, strat_acc, predicted_team_labels, predicted_strat_labels = get_predicted_team_strat(team_num_to_seq_probs, simulated_X, simulated_team_labels, simulated_strategy_labels)

            print('team_acc', team_acc)
            print('predicted_team_labels, simulated_team_labels', [(predicted_team_labels[i], simulated_team_labels[i]) for i in range(len(predicted_team_labels))])

            print('\n\nstrat_acc', strat_acc)

            print('predicted_strat_labels, simulated_strategy_labels',
                  [(predicted_strat_labels[i], simulated_strategy_labels[i]) for i in range(len(predicted_strat_labels))])

            # strat_accuracies.append(np.mean(strat_acc_list))
            # strat_stds.append(np.std(strat_acc_list))

            precision, recall, thresholds = precision_recall_curve(simulated_strategy_labels, predicted_strat_labels)
            strat_acc_list.append(strat_acc)
            strat_recall_list.append(recall)
            strat_prec_list.append(precision)

        strat_acc_means.append(np.mean(strat_acc_list))
        strat_acc_stds.append(np.std(strat_acc_list))

        strat_recall_means.append(np.mean(strat_recall_list))
        strat_recall_stds.append(np.std(strat_recall_list))

        strat_prec_means.append(np.mean(strat_prec_list))
        strat_prec_stds.append(np.std(strat_prec_list))

        # decreasing_max_precision = np.maximum.accumulate(precision[::-1])[::-1]

        # import sklearn.metrics as metrics
        #
        # # calculate the fpr and tpr for all thresholds of the classification
        # # probs = model.predict_proba(X_test)
        # # preds = probs[:, 1]
        # fpr, tpr, threshold = metrics.roc_curve(simulated_strategy_labels, predicted_strat_labels)
        # roc_auc = metrics.auc(fpr, tpr)
        #
        # # method I: plt
        #
        # plt.title('Receiver Operating Characteristic: Window Size = '+str(win_size))
        # plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        # plt.legend(loc='lower right')
        # plt.plot([0, 1], [0, 1], 'r--')
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        # plt.ylabel('True Positive Rate')
        # plt.xlabel('False Positive Rate')
        # plt.savefig('roc_'+str(win_size)+'.png')
        # plt.close()

    # plt.plot(plot_window_size, strat_acc_means)
    # plt.errorbar(plot_window_size, strat_acc_means, yerr=strat_acc_stds, label='STD')
    # plt.xlabel("Window Size")
    # plt.ylabel("Strategy Accuracy")
    # plt.title("Strategy Accuracy vs. Window Size")
    # plt.savefig("strat_acc6.png")
    # plt.close()

    # plt.plot(plot_window_size, strat_recall_means)
    # # plt.errorbar(plot_window_size, strat_recall_means, yerr=strat_recall_stds, label='STD')
    # plt.xlabel("Window Size")
    # plt.ylabel("Strategy Recall")
    # plt.title("Strategy Recall vs. Window Size")
    # plt.savefig("strat_recall6.png")
    # plt.close()

    plt.plot(plot_window_size, strat_prec_means)
    # plt.errorbar(plot_window_size, strat_prec_means, yerr=strat_prec_stds, label='STD')
    plt.xlabel("Window Size")
    plt.ylabel("Strategy Precision")
    plt.title("Strategy Precision vs. Window Size")
    plt.savefig("strat_prec6.png")
    plt.close()

    plt.plot(plot_window_size, strat_acc_means)
    # plt.errorbar(plot_window_size, strat_acc_means, yerr=strat_acc_stds, label='STD')
    plt.xlabel("Window Size")
    plt.ylabel("Strategy Accuracy")
    plt.title("Strategy Accuracy vs. Window Size")
    plt.savefig("strat_acc6.png")
    plt.close()

    plt.plot(plot_window_size, strat_recall_means)
    # plt.errorbar(plot_window_size, strat_recall_means, yerr=strat_recall_stds, label='STD')
    plt.xlabel("Window Size")
    plt.ylabel("Strategy Recall")
    plt.title("Strategy Recall vs. Window Size")
    plt.savefig("strat_recall6.png")
    plt.close()

    plt.scatter(strat_prec_means, strat_recall_means)
    plt.ylabel("Strategy Recall")
    plt.xlabel("Strategy Precision")
    plt.title("Strategy Precision vs. Recall")
    plt.savefig("strat_prec_recall6.png")
    plt.close()

def plot_validation_matrix():
    # num_states_list = [2,3,4]
    # num_clusters_list = [2, 3,4]
    num_states_list = [3]
    num_clusters_list = [2]

    arr = np.zeros((5, 5))

    for n_states in num_states_list:
        test_unsuper_hmm, hidden_seqs, team_numbers, team_num_to_seq_probs = run_naive_hmm_on_p2(n_states=n_states, window=7, ss=3)
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
    plt.savefig('fc_ss_score_1.png')
    plt.close()



if __name__ == '__main__':
    plot_validation_matrix()
    # online_prediction_testing()