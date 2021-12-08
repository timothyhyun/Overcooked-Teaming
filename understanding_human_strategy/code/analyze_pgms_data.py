import json
from process_data import json_eval
from dependencies import *
from hmm import supervised_HMM, unsupervised_HMM, HiddenMarkovModel
from extract_features import *
from sklearn.cluster import AgglomerativeClustering
from extract_features import *
from strategy_hmm_11_29_21 import get_hmm
from human_aware_rl.ppo.ppo import get_ppo_agent, plot_ppo_run, PPO_DATA_DIR
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Action, OvercookedState
from overcooked_ai_py.mdp.actions import Direction, Action
import math

def track_player_actions(layout_params, old_trials, p1_data, p2_data, objects_data, p1_actions,
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
    counter_location_to_id = layout_params["counter_location_to_id"]

    # P2 took what action when one pot had 3 onions? Fill the other or Go for a plate?
    # 0 = Fill the other
    # 1 = Go for a plate
    p2_time_strategic_action = []
    p2_time_strategy = []
    get_next_action = False

    #     p1_private_counters = [4,5]
    #     p2_private_counters = [6,7]
    #     shared_counters = [1,2,3]

    p1_private_counters = layout_params["p1_private_counters"]
    p2_private_counters = layout_params["p2_private_counters"]
    shared_counters = layout_params["shared_counters"]
    onion_dispenser_locations = layout_params["onion_dispenser_locations"]
    dish_dispenser_locations = layout_params["dish_dispenser_locations"]
    serve_locations = layout_params["serve_locations"]
    pot_locations = layout_params["pot_locations"]

    all_counter_locations = list(counter_location_to_id.keys())
    # print("all_counter_locations", all_counter_locations)

    obj_count_id = 0
    next_obj_count_id = 0

    object_list_tracker = {}
    object_location_tracker = {}

    ordered_delivered_tracker = {}

    left_soup_counter_id = 0  # (3,0)
    right_soup_counter_id = 0  # (4,0)

    # left_onion_dispenser_loc = (0, 3)
    # right_onion_dispenser_loc = (1, 4)

    if len(pot_locations) == 2:
        left_pot_loc = pot_locations[0]  # left = top
        right_pot_loc = pot_locations[1]  # right = bottom

    else:
        left_pot_loc = pot_locations[0]
        right_pot_loc = None

    left_soup_contents_dict = {'this_contents': [], 'other_contents': [], 'other_state': 'empty'}  # (3,0)
    right_soup_contents_dict = {'this_contents': [], 'other_contents': [], 'other_state': 'empty'}  # (4,1)
    # states: empty, cooking, cooked, partial

    p1_carrying_soup = None
    p1_carrying_soup_pot_side = None
    p1_carrying_soup_pot_side_id = None
    p1_time_picked_up_soup = None
    p1_time_delivered_soup = None

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

    state_actions_list = []
    hmm_observations = []
    # state_testing_action = 0
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

    """ (10 actions)

            1. P1 pickup onion
            2. P1 put down onion
            3. P1 pick up dish
            4. P1 put down dish
            5. P1 serve soup

            6. P1 pickup onion
            7. P1 put down onion
            8. P1 pick up dish
            9. P1 put down dish
            10. P1 serve soup

        """
    P1_GRAB_ONION_LEFT_D = 0  # 0  - P1 grabs onion from left dispenser
    P1_GRAB_ONION_RIGHT_D = 1  # 1 - P1 grabs onion from right dispenser
    P1_PUT_OBJ_COUNTER = 2  # 2 - P1 puts obj on counter
    P1_GRAB_OBJ_COUNTER = 3  # 3 - P1 grabs obj from counter
    P1_PUT_ONION_LEFT_P = 4  # 4  - P1 puts onion from left pot
    P1_PUT_ONION_RIGHT_P = 5  # 5 - P1 puts onion from right pot
    P1_SERVE = 6  # 6 - P1 serves
    P2_GRAB_ONION_LEFT_D = 7  # 7  - P2 grabs onion from left dispenser
    P2_GRAB_ONION_RIGHT_D = 8  # 8 - P2 grabs onion from right dispenser
    P2_PUT_OBJ_COUNTER = 9  # 9 - P2 puts obj on counter
    P2_GRAB_OBJ_COUNTER = 10  # 10 - P2 grabs obj on counter
    P2_PUT_ONION_LEFT_P = 11  # 11  - P2 puts onion from left pot
    P2_PUT_ONION_RIGHT_P = 12  # 12 - P2 puts onion from right pot
    P2_SERVE = 13  # 13 - P2 serves

    NULL_ACTION = 0
    P1_PICKUP_ONION = 1  # 1. P1 pickup onion
    P1_PUTDOWN_ONION = 2  # 2. P1 put down onion
    P1_PICKUP_DISH = 3  # 3. P1 pick up dish
    P1_PUTDOWN_DISH = 4  # 4. P1 put down dish
    P1_SERVE_SOUP = 5  # 5. P1 serve soup
    #
    P2_PICKUP_ONION = 6  # 6. P2 pickup onion
    P2_PUTDOWN_ONION = 7  # 7. P2 put down onion
    P2_PICKUP_DISH = 8  # 8. P2 pick up dish
    P2_PUTDOWN_DISH = 9  # 9. P2 put down dish
    P2_SERVE_SOUP = 10  # 10. P2 serve soup

    p1_major_action = 17  # initialize as doing nothing (stationary)
    p2_major_action = 17
    # loop over your images
    for a in range(len(t) - 1):
        hmm_action = NULL_ACTION
        # for a in range(100):

        p1_x, p1_y = f_p1(t, a, p1_data)[0], f_p1(t, a, p1_data)[1]
        p1_dir_x, p1_dir_y = arrow_p1(t, a, p1_data)[2], arrow_p1(t, a, p1_data)[3]
        # print('p1_data', p1_data[int(a)])
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

        # print("p2 object info.....", (p2_obj_name, p2_obj_name_next))
        try:
            ################## BEGIN TRACKING PLAYER 1'S MOVEMENT ##################
            # If P1 moves or stays
            if p1_act in ['N', 'S', 'E', 'W']:
                # A1: If P1 is carrying something and moving (dish or onion)
                if players_holding[1] is not None and (p1_obj_x_next is not None and p1_obj_x is not None):
                    obj_location = (p1_obj_x_next, p1_obj_y_next)
                    object_held_id = players_holding[1]

                    prev_location = object_list_tracker[object_held_id]['location']
                    object_list_tracker[object_held_id]['n_actions_since_pickup'] += 1
                    object_list_tracker[object_held_id]['location'] = obj_location
                    object_list_tracker[object_held_id]['p1_n_actions_since_pickup'] += 1

                    # print("object_location_tracker before p1 movement", object_location_tracker)
                    object_location_tracker.pop(prev_location, None)
                    object_location_tracker[obj_location] = object_held_id
                    # print("object_location_tracker after p1 movement", object_location_tracker)

            # If P1 interacted
            if p1_act == 'I' and p1_actions[a] == 'INTERACT':

                # If P1 picked up soup from the pot. P1 would be carrying a dish
                if p1_obj_name_next == 'soup':
                    placed_x, placed_y = list(p1_x_next)[0] + p1_dir_x_next, list(p1_y_next)[0] + p1_dir_y_next
                    # A2: If P1 filled up a dish with the soup
                    if p1_obj_name == 'dish':
                        objects_status = objects_data[a].values()
                        objects_status_next = objects_data[b].values()
                        left_pot_status = 'empty'
                        right_pot_status = 'empty'
                        for obj_status in objects_status:
                            if obj_status['name'] == 'soup' and obj_status['position'] == left_pot_loc:
                                if obj_status['is_cooking'] == True:
                                    left_pot_status = 'cooking'
                                if obj_status['is_ready'] == True:
                                    left_pot_status = 'ready'
                                if obj_status['is_idle'] == True:
                                    left_pot_status = 'idle'
                            if obj_status['name'] == 'soup' and obj_status['position'] == right_pot_loc:
                                if obj_status['is_cooking'] == True:
                                    right_pot_status = 'cooking'
                                if obj_status['is_ready'] == True:
                                    right_pot_status = 'ready'
                                if obj_status['is_idle'] == True:
                                    right_pot_status = 'idle'

                        left_soup_contents_dict['other_state'] = right_pot_status
                        right_soup_contents_dict['other_state'] = left_pot_status

                        p1_time_picked_up_soup = t1

                        obj_location = (list(p1_x)[0], list(p1_y)[0])
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
                        object_list_tracker[obj_count_id]['p1_time_started'] = [t1]
                        object_list_tracker[obj_count_id]['p1_time_completed'] = []
                        object_list_tracker[obj_count_id]['p2_time_started'] = []
                        object_list_tracker[obj_count_id]['p2_time_completed'] = []
                        object_list_tracker[obj_count_id]['both_time_started'] = [
                            t1]  # keep track of absolute time, regardless of contribution
                        object_list_tracker[obj_count_id]['both_time_completed'] = []

                        # Carried or Passed Over Middle
                        object_list_tracker[obj_count_id]['transport_method'] = "carry"

                        # Update what player1 is holding, and increment objects count
                        players_holding[1] = obj_count_id
                        obj_count_id += 1

                        #### CHECK WHICH POT PICKED UP FROM
                        # picked up from left counter
                        if (placed_x, placed_y) == tuple(left_pot_loc):
                            p1_carrying_soup = copy.deepcopy(left_soup_contents_dict)
                            p1_carrying_soup_pot_side = 'left'
                            p1_carrying_soup_pot_side_id = left_soup_counter_id

                            left_soup_contents_dict['this_contents'] = []  # (3,0)
                            right_soup_contents_dict['other_contents'] = []
                            right_soup_contents_dict['other_state'] = 'empty'
                            left_soup_counter_id += 1

                        # picked up from right counter
                        if right_pot_loc is not None and (placed_x, placed_y) == tuple(right_pot_loc):
                            p1_carrying_soup = copy.deepcopy(right_soup_contents_dict)
                            p1_carrying_soup_pot_side = 'right'
                            p1_carrying_soup_pot_side_id = right_soup_counter_id

                            right_soup_contents_dict['this_contents'] = []  # (4,0)
                            left_soup_contents_dict['other_contents'] = []
                            left_soup_contents_dict['other_state'] = 'empty'
                            right_soup_counter_id += 1

                # If P1 picked up an object
                if p1_obj_x is None and p1_obj_x_next is not None:
                    # If P1 picked up soup, skip because we already handled it
                    # if p1_obj_name_next == 'soup':
                    #     continue

                    # P1 picked up an onion or a dish
                    picked_up_from_x, picked_up_from_y = list(p1_x_next)[0] + p1_dir_x_next, list(p1_y_next)[
                        0] + p1_dir_y_next
                    picked_up_loc = (picked_up_from_x, picked_up_from_y)

                    # A3: P1 picked up an onion from the onion dispenser or
                    # A4: P1 picked up an dish from the dish dispenser
                    if (p1_obj_name_next == 'onion' and picked_up_loc in onion_dispenser_locations) or \
                            (p1_obj_name_next == 'dish' and picked_up_loc in dish_dispenser_locations):
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
                        object_list_tracker[obj_count_id]['p1_time_started'] = [t1]
                        object_list_tracker[obj_count_id]['p1_time_completed'] = []
                        object_list_tracker[obj_count_id]['p2_time_started'] = []
                        object_list_tracker[obj_count_id]['p2_time_completed'] = []
                        object_list_tracker[obj_count_id]['both_time_started'] = [
                            t1]  # keep track of absolute time, regardless of contribution
                        object_list_tracker[obj_count_id]['both_time_completed'] = []

                        # Carried or Passed Over Middle
                        object_list_tracker[obj_count_id]['transport_method'] = "carry"

                        # Update what player1 is holding, and increment objects count
                        players_holding[1] = obj_count_id
                        obj_count_id += 1

                        if p1_obj_name_next == 'onion':
                            hmm_action = P1_PICKUP_ONION
                            state_actions_list.append(hmm_action)

                            # if picked_up_loc == left_onion_dispenser_loc:
                            #     hmm_observations.append(P1_GRAB_ONION_LEFT_D)
                            # elif picked_up_loc == right_onion_dispenser_loc:
                            #     hmm_observations.append(P1_GRAB_ONION_RIGHT_D)

                        elif p1_obj_name_next == 'dish':
                            hmm_action = P1_PICKUP_DISH
                            state_actions_list.append(hmm_action)

                    elif picked_up_loc in all_counter_locations:

                        # A5: P1 picked up an onion or dish from the counters
                        hmm_observations.append(P1_GRAB_OBJ_COUNTER)
                        if picked_up_loc in object_location_tracker:
                            obj_picked_id = object_location_tracker[picked_up_loc]
                        else:
                            print('!!! problem p1 pickup not found', picked_up_loc)
                            # print("object_location_tracker", object_location_tracker)
                            if len(object_location_tracker.keys()):
                                # obj_picked_id = 0
                                continue
                            nearest_key = min(list(object_location_tracker.keys()),
                                              key=lambda c: (c[0] - picked_up_from_x) ** 2 + (c[1] - picked_up_from_y) ** 2)


                            # nearest_key = np.random.choice(object_location_tracker.keys())
                            obj_picked_id = object_location_tracker[nearest_key]


                        new_obj_location = (p1_obj_x_next, p1_obj_y_next)

                        object_list_tracker[obj_picked_id]['player_holding'] = 1
                        object_list_tracker[obj_picked_id]['n_actions_since_pickup'] += 1
                        object_list_tracker[obj_picked_id]['location'] = new_obj_location
                        object_list_tracker[obj_picked_id]['on_screen'] = True
                        object_list_tracker[obj_picked_id]['player_holding_list'].append(1)
                        object_list_tracker[obj_picked_id]['p1_n_actions_since_pickup'] += 1
                        object_list_tracker[obj_picked_id]['p1_time_started'].append(t1)

                        players_holding[1] = obj_picked_id

                        object_location_tracker.pop(picked_up_loc, None)
                        object_location_tracker[new_obj_location] = obj_picked_id

                        if p1_obj_name_next == 'onion':
                            hmm_action = P1_PICKUP_ONION
                            state_actions_list.append(hmm_action)
                        elif p1_obj_name_next == 'dish':
                            hmm_action = P1_PICKUP_DISH
                            state_actions_list.append(hmm_action)

                        # If picked-up locations is in the middle counters
                #                     if p1_obj_name_next in ['onion', 'dish'] and picked_up_loc in middle_counter_locations:
                #                         if 1 in object_list_tracker[obj_picked_id]['player_holding_list'] and 2 in object_list_tracker[obj_picked_id]['player_holding_list']:
                #                             if object_list_tracker[obj_picked_id]['transport_method'] == 'middle_pass':
                #                                 record_action_bool = True
                #                                 action_taken = 0

                # If P1 put down an object
                if p1_obj_x is not None and p1_obj_x_next is None:
                    object_held_id = players_holding[1]
                    if object_held_id is None:
                        continue
                    # print("..........................................")
                    # print(".......................P1 object_held_id", object_held_id)
                    # print("put down", ((p1_obj_x, p1_obj_y), p1_obj_x_next, p1_obj_name))
                    # print("..........................................")
                    placed_obj_x, placed_obj_y = list(p1_x_next)[0] + p1_dir_x_next, list(p1_y_next)[0] + p1_dir_y_next
                    placed_location = (placed_obj_x, placed_obj_y)

                    # A6: If P1 put an onion or dish on the counters
                    if p1_obj_name in ['onion', 'dish', 'soup'] and placed_location in all_counter_locations:

                        hmm_observations.append(P1_PUT_OBJ_COUNTER)
                        object_list_tracker[object_held_id]['player_holding'] = 0
                        object_list_tracker[object_held_id]['n_actions_since_pickup'] += 1
                        object_list_tracker[object_held_id]['player_holding_list'].append(0)
                        object_list_tracker[object_held_id]['p1_n_actions_since_pickup'] += 1

                        old_obj_location = object_list_tracker[object_held_id]['location']
                        new_obj_location = (placed_obj_x, placed_obj_y)
                        object_list_tracker[object_held_id]['location'] = new_obj_location

                        # print("p1 object_location_tracker", object_location_tracker)
                        # print("p1 popping old_obj_location", old_obj_location)
                        object_location_tracker.pop(old_obj_location, None)
                        # print("p1 object_location_tracker after", object_location_tracker)
                        # print("p1 putting down obj", new_obj_location)
                        object_location_tracker[new_obj_location] = object_held_id
                        # print("p1 object_location_tracker after new", object_location_tracker)

                        counter_index = counter_location_to_id[(placed_obj_x, placed_obj_y)]
                        object_list_tracker[object_held_id]['counter_used'].append(counter_index)

                        # If the object was placed on a middle counter
                        # if counter_index in middle_counters:
                        #     object_list_tracker[object_held_id]['transport_method'] = 'middle_pass'

                        object_list_tracker[object_held_id]['p1_time_completed'].append(t2)

                        players_holding[1] = None
                        put_down_loc = new_obj_location

                        if p1_obj_name == 'onion':
                            hmm_action = P1_PUTDOWN_ONION
                            state_actions_list.append(hmm_action)
                        elif p1_obj_name == 'dish':
                            hmm_action = P1_PUTDOWN_DISH
                            state_actions_list.append(hmm_action)


                    # A7: If P1 delivered a soup
                    elif p1_obj_name == 'soup' and placed_location in serve_locations:
                        hmm_observations.append(P1_SERVE)
                        p1_time_delivered_soup = t2
                        ordered_delivered_tracker[absolute_order_counter] = {}
                        ordered_delivered_tracker[absolute_order_counter]['details'] = p1_carrying_soup
                        ordered_delivered_tracker[absolute_order_counter]['pot_side'] = p1_carrying_soup_pot_side
                        ordered_delivered_tracker[absolute_order_counter]['pot_side_id'] = p1_carrying_soup_pot_side_id
                        ordered_delivered_tracker[absolute_order_counter]['time_picked_up'] = p1_time_picked_up_soup
                        ordered_delivered_tracker[absolute_order_counter]['time_delivered'] = p1_time_delivered_soup

                        absolute_order_counter += 1
                        players_holding[1] = None  # Comment out maybe?

                        state_actions_list.append(P1_SERVE_SOUP)

                        object_list_tracker[object_held_id]['player_holding'] = 0
                        object_list_tracker[object_held_id]['n_actions_since_pickup'] += 1
                        object_list_tracker[object_held_id]['player_holding_list'].append(0)
                        object_list_tracker[object_held_id]['p1_n_actions_since_pickup'] += 1

                        old_obj_location = object_list_tracker[object_held_id]['location']
                        new_obj_location = (placed_obj_x, placed_obj_y)
                        object_list_tracker[object_held_id]['location'] = new_obj_location

                        # print("object_location_tracker before check pots", object_location_tracker)
                        object_location_tracker.pop(old_obj_location, None)
                        # object_location_tracker[new_obj_location] = object_held_id
                        # print("object_location_tracker after check pots", object_location_tracker)

                        players_holding[1] = None


                    # Else, A8: P1 placed an onion in one of the pots
                    elif p1_obj_name == 'onion':

                        # print(f" A8: P1 placed an onion in one of the pots, object_held_id: {object_held_id}")

                        # placed at left counter pot
                        if placed_location == tuple(left_pot_loc):
                            state_actions_list.append(P1_PUTDOWN_ONION)
                            hmm_observations.append(P1_PUT_ONION_LEFT_P)
                            left_soup_contents_dict['this_contents'].append(object_held_id)
                            right_soup_contents_dict['other_contents'].append(object_held_id)

                        # placed at right counter pot
                        if right_pot_loc is not None and placed_location == tuple(right_pot_loc):
                            state_actions_list.append(P1_PUTDOWN_ONION)
                            hmm_observations.append(P1_PUT_ONION_RIGHT_P)
                            right_soup_contents_dict['this_contents'].append(object_held_id)
                            left_soup_contents_dict['other_contents'].append(object_held_id)

                        if placed_location == tuple(left_pot_loc) or (
                                right_pot_loc is not None and placed_location == tuple(right_pot_loc)):
                            object_list_tracker[object_held_id]['player_holding'] = 0
                            object_list_tracker[object_held_id]['n_actions_since_pickup'] += 1
                            object_list_tracker[object_held_id]['player_holding_list'].append(0)
                            object_list_tracker[object_held_id]['p1_n_actions_since_pickup'] += 1

                            old_obj_location = object_list_tracker[object_held_id]['location']
                            new_obj_location = (placed_obj_x, placed_obj_y)
                            object_list_tracker[object_held_id]['location'] = new_obj_location

                            # print("object_location_tracker before check pots", object_location_tracker)
                            object_location_tracker.pop(old_obj_location, None)
                            object_location_tracker[new_obj_location] = object_held_id
                            # print("object_location_tracker after check pots", object_location_tracker)

                            players_holding[1] = None

                        # P1 placed object on counter
                        if (placed_obj_x, placed_obj_y) in counter_location_to_id:
                            # state_actions_list.append(P1_PUTDOWN_ONION)
                            hmm_observations.append(P1_PUT_OBJ_COUNTER)
                            counter_index = counter_location_to_id[(placed_obj_x, placed_obj_y)]
                            object_list_tracker[object_held_id]['counter_used'].append(counter_index)
                            players_holding[1] = None
                            # if counter_index in middle_counters:
                            #     object_list_tracker[object_held_id]['transport_method'] = 'middle_pass'

                        object_list_tracker[object_held_id]['p1_time_completed'].append(t2)

                        # If onion is placed in pot, check which action should be appended
                        # if p1_obj_name == 'onion' and placed_location in [tuple(left_pot_loc), tuple(right_pot_loc)]:
                        #     # if 1 in object_list_tracker[object_held_id]['player_holding_list'] and 2 in \
                        #     #         object_list_tracker[object_held_id]['player_holding_list']:
                        #     if object_list_tracker[object_held_id]['transport_method'] == 'middle_pass':
                        #         record_action_bool = True
                        #         action_taken = 0
                        #     elif object_list_tracker[object_held_id]['transport_method'] == 'carry':
                        #         record_action_bool = True
                        #         action_taken = 1

                    # else:

            ################## END OF PLAYER 1'S MOVEMENT ##################

            ################## BEGIN TRACKNG PLAYER 2'S MOVEMENT ##################
            # If P2 moves or stays
            if p2_act in ['N', 'S', 'E', 'W']:
                # A1: If P2 is carrying something and moving (dish or onion)
                if players_holding[2] is not None and (p2_obj_x_next is not None and p2_obj_x is not None):
                    obj_location = (p2_obj_x_next, p2_obj_y_next)
                    object_held_id = players_holding[2]

                    prev_location = object_list_tracker[object_held_id]['location']
                    object_list_tracker[object_held_id]['n_actions_since_pickup'] += 1
                    object_list_tracker[object_held_id]['location'] = obj_location
                    object_list_tracker[object_held_id]['p1_n_actions_since_pickup'] += 1

                    # print("object_location_tracker before p2 movement", object_location_tracker)
                    object_location_tracker.pop(prev_location, None)
                    object_location_tracker[obj_location] = object_held_id
                    # print("object_location_tracker after p2 movement", object_location_tracker)

            # If P2 interacted
            if p2_act == 'I' and p2_actions[a] == 'INTERACT':

                # If P2 picked up soup from the pot. P2 would be carrying a dish
                if p2_obj_name_next == 'soup':
                    placed_x, placed_y = list(p2_x_next)[0] + p2_dir_x_next, list(p2_y_next)[0] + p2_dir_y_next
                    # A2: If P2 filled up a dish with the soup
                    if p2_obj_name == 'dish':
                        # print("p2 picked up soup")
                        objects_status = objects_data[a].values()
                        objects_status_next = objects_data[b].values()
                        left_pot_status = 'empty'
                        right_pot_status = 'empty'
                        print("objects_status", objects_status)
                        for obj_status in objects_status:
                            if obj_status['name'] == 'soup' and obj_status['position'] == left_pot_loc:
                                if obj_status['is_cooking'] == True:
                                    left_pot_status = 'cooking'
                                if obj_status['is_ready'] == True:
                                    left_pot_status = 'ready'
                                if obj_status['is_idle'] == True:
                                    left_pot_status = 'idle'
                            if obj_status['name'] == 'soup' and obj_status['position'] == right_pot_loc:
                                if obj_status['is_cooking'] == True:
                                    right_pot_status = 'cooking'
                                if obj_status['is_ready'] == True:
                                    right_pot_status = 'ready'
                                if obj_status['is_idle'] == True:
                                    right_pot_status = 'idle'

                        left_soup_contents_dict['other_state'] = right_pot_status
                        right_soup_contents_dict['other_state'] = left_pot_status

                        p2_time_picked_up_soup = t1

                        obj_location = (list(p2_x)[0], list(p2_y)[0])
                        object_location_tracker[obj_location] = obj_count_id
                        if obj_count_id not in object_list_tracker:
                            object_list_tracker[obj_count_id] = {}
                        object_list_tracker[obj_count_id]['name'] = p2_obj_name_next
                        object_list_tracker[obj_count_id]['player_holding'] = 2
                        object_list_tracker[obj_count_id]['id'] = obj_count_id
                        object_list_tracker[obj_count_id]['n_actions_since_pickup'] = 0
                        object_list_tracker[obj_count_id]['location'] = obj_location
                        object_list_tracker[obj_count_id]['on_screen'] = True
                        object_list_tracker[obj_count_id]['player_holding_list'] = [2]
                        object_list_tracker[obj_count_id]['p1_n_actions_since_pickup'] = 0
                        object_list_tracker[obj_count_id]['p2_n_actions_since_pickup'] = 1
                        object_list_tracker[obj_count_id]['counter_used'] = []
                        object_list_tracker[obj_count_id]['p1_time_started'] = []
                        object_list_tracker[obj_count_id]['p1_time_completed'] = []
                        object_list_tracker[obj_count_id]['p2_time_started'] = [t1]
                        object_list_tracker[obj_count_id]['p2_time_completed'] = []
                        object_list_tracker[obj_count_id]['both_time_started'] = [
                            t1]  # keep track of absolute time, regardless of contribution
                        object_list_tracker[obj_count_id]['both_time_completed'] = []

                        # Carried or Passed Over Middle
                        object_list_tracker[obj_count_id]['transport_method'] = "carry"

                        # Update what player1 is holding, and increment objects count
                        players_holding[2] = obj_count_id
                        obj_count_id += 1

                        #### CHECK WHICH POT PICKED UP FROM
                        # picked up from left counter
                        if (placed_x, placed_y) == tuple(left_pot_loc):
                            p2_carrying_soup = copy.deepcopy(left_soup_contents_dict)
                            p2_carrying_soup_pot_side = 'left'
                            p2_carrying_soup_pot_side_id = left_soup_counter_id

                            left_soup_contents_dict['this_contents'] = []  # (3,0)
                            right_soup_contents_dict['other_contents'] = []
                            right_soup_contents_dict['other_state'] = 'empty'
                            left_soup_counter_id += 1

                        # picked up from right counter
                        if right_pot_loc is not None and (placed_x, placed_y) == tuple(right_pot_loc):
                            p2_carrying_soup = copy.deepcopy(right_soup_contents_dict)
                            p2_carrying_soup_pot_side = 'right'
                            p2_carrying_soup_pot_side_id = right_soup_counter_id

                            right_soup_contents_dict['this_contents'] = []  # (4,0)
                            left_soup_contents_dict['other_contents'] = []
                            left_soup_contents_dict['other_state'] = 'empty'
                            right_soup_counter_id += 1

                # If P2 picked up an object
                if p2_obj_x is None and p2_obj_x_next is not None:

                    # If P2 picked up soup, skip because we already handled it
                    # if p2_obj_name_next == 'soup':
                    #     continue

                    # P2 picked up an onion or a dish
                    picked_up_from_x, picked_up_from_y = list(p2_x_next)[0] + p2_dir_x_next, list(p2_y_next)[
                        0] + p2_dir_y_next
                    picked_up_loc = (picked_up_from_x, picked_up_from_y)

                    # A3: P2 picked up an onion from the onion dispenser or
                    # A4: P2 picked up an dish from the dish dispenser
                    if (p2_obj_name_next == 'onion' and picked_up_loc in onion_dispenser_locations) or \
                            (p2_obj_name_next == 'dish' and picked_up_loc in dish_dispenser_locations):
                        obj_location = (p2_obj_x_next, p2_obj_y_next)
                        object_location_tracker[obj_location] = obj_count_id

                        if obj_count_id not in object_list_tracker:
                            object_list_tracker[obj_count_id] = {}
                        object_list_tracker[obj_count_id]['name'] = p2_obj_name_next
                        object_list_tracker[obj_count_id]['player_holding'] = 2
                        object_list_tracker[obj_count_id]['id'] = obj_count_id
                        object_list_tracker[obj_count_id]['n_actions_since_pickup'] = 0
                        object_list_tracker[obj_count_id]['location'] = obj_location
                        object_list_tracker[obj_count_id]['on_screen'] = True
                        object_list_tracker[obj_count_id]['player_holding_list'] = [2]
                        object_list_tracker[obj_count_id]['p1_n_actions_since_pickup'] = 0
                        object_list_tracker[obj_count_id]['p2_n_actions_since_pickup'] = 1
                        object_list_tracker[obj_count_id]['counter_used'] = []
                        object_list_tracker[obj_count_id]['p1_time_started'] = []
                        object_list_tracker[obj_count_id]['p1_time_completed'] = []
                        object_list_tracker[obj_count_id]['p2_time_started'] = [t1]
                        object_list_tracker[obj_count_id]['p2_time_completed'] = []
                        object_list_tracker[obj_count_id]['both_time_started'] = [
                            t1]  # keep track of absolute time, regardless of contribution
                        object_list_tracker[obj_count_id]['both_time_completed'] = []

                        # Carried or Passed Over Middle
                        object_list_tracker[obj_count_id]['transport_method'] = "carry"

                        # Update what player1 is holding, and increment objects count
                        players_holding[2] = obj_count_id
                        obj_count_id += 1

                        if p2_obj_name_next == 'onion':
                            hmm_action = P2_PICKUP_ONION
                            state_actions_list.append(hmm_action)
                            # if picked_up_loc == left_onion_dispenser_loc:
                            #     hmm_observations.append(P2_GRAB_ONION_LEFT_D)
                            # elif picked_up_loc == right_onion_dispenser_loc:
                            #     hmm_observations.append(P2_GRAB_ONION_RIGHT_D)
                        elif p2_obj_name_next == 'dish':
                            hmm_action = P2_PICKUP_DISH
                            state_actions_list.append(hmm_action)




                    elif picked_up_loc in all_counter_locations:
                        hmm_observations.append(P2_GRAB_OBJ_COUNTER)
                        # A5: P2 picked up an onion or dish from the counters
                        if picked_up_loc in object_location_tracker:
                            obj_picked_id = object_location_tracker[picked_up_loc]
                        else:
                            print('!!! problem p2 pickup not found')
                            # print("p2_Act", p2_act)
                            # print("p2_actions[a]", p2_actions[a])
                            # print("p2_actions[b]", p2_actions[b])
                            nearest_key = min(list(object_location_tracker.keys()),
                                              key=lambda c: (c[0] - placed_x) ** 2 + (c[1] - placed_y) ** 2)
                            obj_picked_id = object_location_tracker[nearest_key]

                        new_obj_location = (p2_obj_x_next, p2_obj_y_next)

                        object_list_tracker[obj_picked_id]['player_holding'] = 2
                        object_list_tracker[obj_picked_id]['n_actions_since_pickup'] += 1
                        object_list_tracker[obj_picked_id]['location'] = new_obj_location
                        object_list_tracker[obj_picked_id]['on_screen'] = True
                        object_list_tracker[obj_picked_id]['player_holding_list'].append(1)
                        object_list_tracker[obj_picked_id]['p2_n_actions_since_pickup'] += 1
                        object_list_tracker[obj_picked_id]['p2_time_started'].append(t1)

                        players_holding[2] = obj_picked_id

                        # print("object_location_tracker before p2 obj pikcup", object_location_tracker)
                        object_location_tracker.pop(picked_up_loc, None)
                        object_location_tracker[new_obj_location] = obj_picked_id
                        # print("object_location_tracker after p2 obj pikcup", object_location_tracker)

                        if p2_obj_name_next == 'onion':
                            hmm_action = P2_PICKUP_ONION
                            state_actions_list.append(hmm_action)
                        elif p2_obj_name_next == 'dish':
                            hmm_action = P2_PICKUP_DISH
                            state_actions_list.append(hmm_action)

                        # If picked-up locations is in the middle counters
                #                     if p1_obj_name_next in ['onion', 'dish'] and picked_up_loc in middle_counter_locations:
                #                         if 1 in object_list_tracker[obj_picked_id]['player_holding_list'] and 2 in object_list_tracker[obj_picked_id]['player_holding_list']:
                #                             if object_list_tracker[obj_picked_id]['transport_method'] == 'middle_pass':
                #                                 record_action_bool = True
                #                                 action_taken = 0

                # If P2 put down an object
                if p2_obj_x is not None and p2_obj_x_next is None:
                    object_held_id = players_holding[2]
                    # print("players_holding", players_holding)
                    placed_obj_x, placed_obj_y = list(p2_x_next)[0] + p2_dir_x_next, list(p2_y_next)[0] + p2_dir_y_next
                    placed_location = (placed_obj_x, placed_obj_y)

                    # A6: If P2 put an onion or dish on the counters
                    if p2_obj_name in ['onion', 'dish', 'soup'] and placed_location in all_counter_locations:
                        hmm_observations.append(P2_PUT_OBJ_COUNTER)

                        object_list_tracker[object_held_id]['player_holding'] = 0
                        object_list_tracker[object_held_id]['n_actions_since_pickup'] += 1
                        object_list_tracker[object_held_id]['player_holding_list'].append(0)
                        object_list_tracker[object_held_id]['p2_n_actions_since_pickup'] += 1

                        old_obj_location = object_list_tracker[object_held_id]['location']
                        new_obj_location = (placed_obj_x, placed_obj_y)
                        object_list_tracker[object_held_id]['location'] = new_obj_location

                        object_location_tracker.pop(old_obj_location, None)
                        object_location_tracker[new_obj_location] = object_held_id

                        # print("p2 object_location_tracker", object_location_tracker)
                        # print("p2 popping old_obj_location", old_obj_location)
                        object_location_tracker.pop(old_obj_location, None)
                        # print("p2 object_location_tracker after", object_location_tracker)
                        # print("p2 putting down obj", new_obj_location)
                        object_location_tracker[new_obj_location] = object_held_id
                        # print("p2 object_location_tracker after put down on counter", object_location_tracker)

                        counter_index = counter_location_to_id[(placed_obj_x, placed_obj_y)]
                        object_list_tracker[object_held_id]['counter_used'].append(counter_index)

                        # If the object was placed on a middle counter
                        # if counter_index in middle_counters:
                        #     object_list_tracker[object_held_id]['transport_method'] = 'middle_pass'

                        object_list_tracker[object_held_id]['p2_time_completed'].append(t2)

                        players_holding[2] = None
                        put_down_loc = new_obj_location

                        if p2_obj_name == 'onion':
                            hmm_action = P2_PUTDOWN_ONION
                            state_actions_list.append(hmm_action)
                        elif p2_obj_name == 'dish':
                            hmm_action = P2_PUTDOWN_DISH
                            state_actions_list.append(hmm_action)


                    # A7: If P2 delivered a soup
                    elif p2_obj_name == 'soup' and placed_location in serve_locations:
                        p2_time_delivered_soup = t2
                        ordered_delivered_tracker[absolute_order_counter] = {}
                        ordered_delivered_tracker[absolute_order_counter]['details'] = p2_carrying_soup
                        ordered_delivered_tracker[absolute_order_counter]['pot_side'] = p2_carrying_soup_pot_side
                        ordered_delivered_tracker[absolute_order_counter]['pot_side_id'] = p2_carrying_soup_pot_side_id
                        ordered_delivered_tracker[absolute_order_counter]['time_picked_up'] = p2_time_picked_up_soup
                        ordered_delivered_tracker[absolute_order_counter]['time_delivered'] = p2_time_delivered_soup

                        absolute_order_counter += 1

                        hmm_observations.append(P2_SERVE)

                        state_actions_list.append(P2_SERVE_SOUP)
                        # print("p2_obj_name", p2_obj_name)
                        # print("players_holding", players_holding)
                        object_list_tracker[object_held_id]['player_holding'] = 0
                        object_list_tracker[object_held_id]['n_actions_since_pickup'] += 1
                        object_list_tracker[object_held_id]['player_holding_list'].append(0)
                        object_list_tracker[object_held_id]['p2_n_actions_since_pickup'] += 1

                        old_obj_location = object_list_tracker[object_held_id]['location']
                        new_obj_location = (placed_obj_x, placed_obj_y)
                        object_list_tracker[object_held_id]['location'] = new_obj_location

                        # print("object_location_tracker before p2 pot check", object_location_tracker)
                        object_location_tracker.pop(old_obj_location, None)
                        # object_location_tracker[new_obj_location] = object_held_id
                        # print("object_location_tracker after p2 pot check", object_location_tracker)
                        players_holding[2] = None

                    # Else, A8: P2 placed an onion in one of the pots
                    elif p2_obj_name == 'onion':

                        # placed at left counter pot
                        if placed_location == tuple(left_pot_loc):
                            state_actions_list.append(P2_PUTDOWN_ONION)
                            hmm_observations.append(P2_PUT_ONION_LEFT_P)
                            left_soup_contents_dict['this_contents'].append(object_held_id)
                            right_soup_contents_dict['other_contents'].append(object_held_id)

                        # placed at right counter pot
                        if right_pot_loc is not None and placed_location == tuple(right_pot_loc):
                            state_actions_list.append(P2_PUTDOWN_ONION)
                            hmm_observations.append(P2_PUT_ONION_RIGHT_P)
                            right_soup_contents_dict['this_contents'].append(object_held_id)
                            left_soup_contents_dict['other_contents'].append(object_held_id)

                        if placed_location == tuple(left_pot_loc) or (
                                right_pot_loc is not None and placed_location == tuple(right_pot_loc)):
                            object_list_tracker[object_held_id]['player_holding'] = 0
                            object_list_tracker[object_held_id]['n_actions_since_pickup'] += 1
                            object_list_tracker[object_held_id]['player_holding_list'].append(0)
                            object_list_tracker[object_held_id]['p2_n_actions_since_pickup'] += 1

                            old_obj_location = object_list_tracker[object_held_id]['location']
                            new_obj_location = (placed_obj_x, placed_obj_y)
                            object_list_tracker[object_held_id]['location'] = new_obj_location

                            # print("object_location_tracker before p2 pot check", object_location_tracker)
                            object_location_tracker.pop(old_obj_location, None)
                            # object_location_tracker[new_obj_location] = object_held_id
                            # print("object_location_tracker after p2 pot check", object_location_tracker)
                            players_holding[2] = None

                        if (placed_obj_x, placed_obj_y) in counter_location_to_id:
                            # state_actions_list.append(P2_PUTDOWN_ONION)
                            hmm_observations.append(P2_PUT_OBJ_COUNTER)
                            counter_index = counter_location_to_id[(placed_obj_x, placed_obj_y)]
                            object_list_tracker[object_held_id]['counter_used'].append(counter_index)
                            players_holding[2] = None
                            # if counter_index in middle_counters:
                            #     object_list_tracker[object_held_id]['transport_method'] = 'middle_pass'

                        object_list_tracker[object_held_id]['p2_time_completed'].append(t2)

                        # If onion is placed in pot, check which action should be appended
                        # if p2_obj_name == 'onion' and placed_location in [tuple(left_pot_loc), tuple(right_pot_loc)]:
                        #     # if 1 in object_list_tracker[object_held_id]['player_holding_list'] and 2 in \
                        #     #         object_list_tracker[object_held_id]['player_holding_list']:
                        #     if object_list_tracker[object_held_id]['transport_method'] == 'middle_pass':
                        #         record_action_bool = True
                        #         action_taken = 0
                        #     elif object_list_tracker[object_held_id]['transport_method'] == 'carry':
                        #         record_action_bool = True
                        #         action_taken = 2

                    # else:
                    #     print("\n\nwhat other case, p2")
                    #     print("p1_data: ", p1_data[a])
                    #     print("p2_data: ", p2_data[a])
                    #     print("p1_actions: ", p1_actions[a])
                    #     print("p2_actions: ", p2_actions[a])
                    #     print("objects_data: ", objects_data[a])
                    #     print("\nNEXT STEP")
                    #     print("p1_data: ", p1_data[b])
                    #     print("p2_data: ", p2_data[b])
                    #     print("p1_actions: ", p1_actions[b])
                    #     print("p2_actions: ", p2_actions[b])
                    #     print("objects_data: ", objects_data[b])
                    #     print()

                    ################## END OF PLAYER 2'S MOVEMENT ##################
        except:
            continue

        ##### SAVE TO LISTS
        # if record_action_bool == True:
        #     hmm_observations.append(action_taken)
        # print(hmm_observations)

    # state_actions_list_new = []
    # remove_actions = [1,2,3,4,5,7,9,8]
    # # remove_actions = []
    # for elem in state_actions_list:
    #     if elem not in remove_actions:
    #         state_actions_list_new.append(elem)
    # state_actions_list = state_actions_list_new

    return object_list_tracker, ordered_delivered_tracker, hmm_observations, state_actions_list


def featurize_data_for_naive_hmm(layout_trials, layout_name, layout_params, window=20, ss=10):
    team_chunked_actions_data = {}

    name = layout_name
    title = layout_name

    old_trials = import_2019_data()

    team_numbers = []
    # print("......Loading trials", layout_trials)
    for trial_id in layout_trials:
        # for j in [5]:
        # trial_id = layout_trials[j]
        team_numbers.append(trial_id)

        state_data = layout_trials[trial_id]['state_data']
        joint_actions = layout_trials[trial_id]['joint_action']
        time_elapsed = layout_trials[trial_id]['time_elapsed']

        p1_data = []
        p2_data = []
        p1_actions = []
        p2_actions = []
        state_data_eval = []
        objects_data = []
        for i in range(1, len(state_data)):
            # prev_state_x = json_eval(state_data[i - 1])
            # state_x = json_eval(state_data[i])
            # joint_actions_i = literal_eval(joint_actions[i])
            prev_state_x = (state_data[i - 1]['state'])
            state_x = (state_data[i]['state'])
            joint_actions_i = (joint_actions[i])

            p1_index = 1
            p2_index = 0

            p1_data.append(state_x['players'][p1_index])
            p2_data.append(state_x['players'][p2_index])
            state_data_eval.append(state_x)
            objects_data.append(state_x['objects'])

            p1_actions.append(joint_actions_i[p1_index])
            p2_actions.append(joint_actions_i[p2_index])

        object_list_tracker, ordered_delivered_tracker, hmm_observations, state_actions_list = track_player_actions(
            layout_params,old_trials,
            p1_data, p2_data, objects_data, p1_actions,
            p2_actions, name, time_elapsed)
        team_chunked_actions_data[trial_id] = {}
        # team_chunked_actions_data[trial_id]['p1_actions'] = p1_actions_list
        # team_chunked_actions_data[trial_id]['p2_actions'] = p2_actions_list
        # team_chunked_actions_data[trial_id]['p2_major_actions'] = p2_time_strategic_action
        # team_chunked_actions_data[trial_id]['p2_delivery_times'] = p2_time_strategy
        team_chunked_actions_data[trial_id]['state_actions_list'] = state_actions_list
        print('team', trial_id)
        print('state_actions_list', state_actions_list)
    #     other_pot_contains_num_onions, other_pot_states, steps_p1_took_total_order, \
    #         steps_p2_took_total_order, order_completion_times = pull_features_from_output(object_list_tracker, ordered_delivered_tracker)

    #     plot_results(other_pot_contains_num_onions, other_pot_states, steps_p1_took_total_order, steps_p2_took_total_order, order_completion_times)
    #     print()
    # team_idx = 9

    X_data = {}
    X_data_chunked, team_numbers_chunked = [], []
    X_data = []
    for team_idx in team_chunked_actions_data:

        state_actions_list = team_chunked_actions_data[team_idx]['state_actions_list']
        add = []
        for j in range(len(state_actions_list)):
            add.append(state_actions_list[j] - 1)
        X_data.append(np.array(add))

        #     print(len(p2_major_actions))
        # add = []
        # for j in range(5):
        #     add.append(p2_major_actions[j][1])

        # observation_list = team_chunked_actions_data[team_idx]['state_actions_list']
        # add = [observation_list[c] for c in range(len(observation_list))]
        # X_data.append(np.array(add))
        # team_numbers.append(team_idx)
        #
        # for j in range(0, len(observation_list) - window, ss):
        #     add = [observation_list[c] for c in range(j, j + window)]
        #     X_data_chunked.append(np.array(add))
        #     team_numbers_chunked.append(team_idx)

        # X_data = np.array(X_data)
        # print(f'Team: {team_idx}, add = {(np.unique(add))}')
    # X_data, team_numbers = X_data_chunked, team_numbers_chunked
    return X_data, team_numbers


def featurize_data_for_naive_hmm_w_window(window=4, ss=3):
    team_chunked_actions_data = {}

    name = 'random3'
    title = 'Counter Circuit'

    # name = 'random0'
    # title = 'Forced Coordination'
    old_trials = import_2019_data()
    layout_trials = old_trials[old_trials['layout_name'] == name]['trial_id'].unique()
    # name = 'random0'
    # title = 'Forced Coordination'
    trial_data = {}

    for j in range(len(layout_trials)):
        # for j in [5]:
        trial_id = layout_trials[j]
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

        object_list_tracker, ordered_delivered_tracker, hmm_observations = track_player_actions(old_trials, p1_data,
                                                                                                p2_data,
                                                                                                objects_data,
                                                                                                p1_actions,
                                                                                                p2_actions, name,
                                                                                                time_elapsed)
        # print("hmm_observations", hmm_observations)
        team_chunked_actions_data[trial_id] = {}
        team_chunked_actions_data[trial_id]['hmm_observations'] = hmm_observations

    # X_data = []
    # team_numbers = []
    # for team_idx in team_chunked_actions_data:
    #     add = team_chunked_actions_data[team_idx]['hmm_observations']
    #     print('len add', len(add))
    #     X_data.append(np.array(add))
    #     team_numbers.append(team_idx)

    # X_data = np.array(X_data)

    # return X_data, team_numbers

    X_data = []
    team_numbers = []

    X_data_chunked = []
    team_numbers_chunked = []

    for team_idx in team_chunked_actions_data:

        observation_list = team_chunked_actions_data[team_idx]['hmm_observations']

        #     print(len(p2_major_actions))
        # add = []
        # for j in range(5):
        #     add.append(p2_major_actions[j][1])

        # add = []
        add = [observation_list[c] for c in range(len(observation_list))]
        X_data.append(np.array(add))
        team_numbers.append(team_idx)

        for j in range(0, len(observation_list) - window, ss):
            add = [observation_list[c] for c in range(j, j + window)]
            X_data_chunked.append(np.array(add))
            team_numbers_chunked.append(team_idx)

    # X_data = np.array(X_data)
    # return X_data, team_numbers
    return X_data, team_numbers, X_data_chunked, team_numbers_chunked


def run_naive_hmm_on_p2_method_2(n_states, window=4, ss=2):
    # X = observation_data
    # Y = hidden_state_data
    # X, team_numbers, X_data_chunked, team_numbers_chunked = featurize_data_for_naive_hmm(window=window, ss=ss)
    X, team_numbers = featurize_data_for_naive_hmm()

    N_iters = 100

    test_unsuper_hmm = unsupervised_HMM(X, n_states, N_iters)

    # print('emission', test_unsuper_hmm.generate_emission(10))
    hidden_seqs = []
    team_num_to_seq_probs = {}
    for j in range(len(X)):
        trial_data = X[j][:100]
        viterbi_output, all_sequences_and_probs = test_unsuper_hmm.viterbi_all_probs(trial_data)
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


def cut_to_same_length(hidden_seqs):
    min_len = min(len(elem) for elem in hidden_seqs)
    X = []
    for i in range(len(hidden_seqs)):
        team_hs = hidden_seqs[i]
        team_hs_pad = []
        for j in range(min_len):
            if j < len(team_hs):
                team_hs_pad.append(team_hs[j + len(team_hs) - min_len])
            # else:
            #     team_hs_pad.append(team_hs_mode)
        X.append(team_hs_pad)
    # print('X', X)
    return np.array(X)



def cluster_hidden_states(hidden_seqs, n_clusters=2):
    X = pad_w_mode(hidden_seqs)
    # X = cut_to_same_length(hidden_seqs)
    # print('X=', X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    ss = sklearn.metrics.silhouette_score(X, cluster_labels)
    # ss = sklearn.metrics.calinski_harabasz_score(X, cluster_labels)
    return cluster_labels, cluster_centers, ss


def cluster_hidden_states_new(hidden_seqs, n_clusters=2):
    X = pad_w_mode(hidden_seqs)
    # print('X=', X)
    # X = process_no_pad(hidden_seqs)
    # X = cut_to_same_length(hidden_seqs)
    acluster = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
    cluster_labels = acluster.labels_
    ss = sklearn.metrics.silhouette_score(X, cluster_labels)
    # cluster_centers = acluster.cluster_centers_
    cluster_centers = "None"
    return cluster_labels, cluster_centers, ss


def reindex_state_tuples(team_state_data):
    all_state_tuples = []
    # for trial_id in team_state_data:
    for trial_id in range(len(team_state_data)):
        all_state_tuples.extend(list(team_state_data[trial_id]))

    # print('all_state_tuples', all_state_tuples)
    unique_state_tuples = sorted(list(set(all_state_tuples)))
    idx_to_state_tuple = dict((i, j) for i, j in enumerate(unique_state_tuples))
    state_tuple_to_idx = dict((j, i) for i, j in enumerate(unique_state_tuples))

    new_team_state_data = []
    for trial_id in range(len(team_state_data)):
        new_team_state_data.append(np.array([int(state_tuple_to_idx[state]) for state in team_state_data[trial_id]]))

    return new_team_state_data


def run_naive_hmm_on_p2(layout_trials, layout_name, layout_params, n_states, window=4, ss=2):
    # X = observation_data
    # Y = hidden_state_data
    # X, team_numbers, X_data_chunked, team_numbers_chunked = featurize_data_for_naive_hmm_w_window(window=window, ss=ss)
    X_dict, team_numbers = featurize_data_for_naive_hmm(layout_trials, layout_name, layout_params)

    X = reindex_state_tuples(X_dict)
    # print('X', X)
    N_iters = 100
    print("RUNNING HMM!!!!.............")
    # test_unsuper_hmm = unsupervised_HMM(X, n_states, N_iters)
    test_unsuper_hmm = get_hmm(layout_name, n_states)
    print("DONE RUNNING HMM!!!!.............")

    # print('emission', test_unsuper_hmm.generate_emission(10))
    hidden_seqs = []
    team_num_to_seq_probs = {}
    for j in range(len(X)):
        # print("team", team_numbers[j])
        # print("reindex", X[j][:50])
        viterbi_output, all_sequences_and_probs = test_unsuper_hmm.viterbi_all_probs(X[j])
        team_num_to_seq_probs[team_numbers[j]] = all_sequences_and_probs
        hidden_seqs.append([int(x) for x in viterbi_output])
        print('viterbi: hidden seq: Team ' + str(team_numbers[j]) + ": ", viterbi_output)

    return test_unsuper_hmm, hidden_seqs, team_numbers, team_num_to_seq_probs


def get_layout_params(layout_name):
    layout_params_dict = {
        'random0': {
            "counter_location_to_id": {
                (2, 1): 1,
                (2, 2): 2,
                (2, 3): 3,
                (1, 0): 4,
                (1, 4): 5,
                (4, 2): 6,
                (4, 3): 7
            },
            "p1_private_counters": [(1, 0), (1, 4)],
            "p2_private_counters": [(4, 2), (4, 3)],
            "shared_counters": [(2, 1), (2, 2), (2, 3)],
            "onion_dispenser_locations": [(0, 1), (0, 2)],
            "dish_dispenser_locations": [(0, 3)],
            "pot_locations": [(3, 0), (4, 1)],
            "serve_locations": [(3, 4)],
        },
        'random3': {
            "counter_location_to_id": {
                (0, 1): 1,
                (1, 0): 2,
                (2, 0): 3,
                (5, 0): 4,
                (6, 0): 5,
                (7, 1): 6,
                (7, 3): 7,
                (6, 4): 8,
                (5, 4): 9,
                (2, 4): 10,
                (1, 4): 11,
                (0, 3): 12,
                (2, 2): 13,
                (3, 2): 14,
                (4, 2): 15,
                (5, 2): 16,
            },
            "p1_private_counters": [],
            "p2_private_counters": [],
            "shared_counters": [],
            "onion_dispenser_locations": [(3, 4), (4, 4)],
            "dish_dispenser_locations": [(0, 2)],
            "pot_locations": [(3, 0), (4, 0)],
            "serve_locations": [(7, 2)],

        },
        'asymmetric_advantages': {
            "counter_location_to_id": {
                (0, 3): 1,
                (0, 2): 2,
                (1, 0): 3,
                (2, 1): 4,
                (1, 4): 5,
                (2, 4): 6,
                (6, 1): 7,
                (7, 0): 8,
                (8, 2): 9,
                (8, 3): 10,
                (7, 4): 11,
                (6, 4): 12

            },
            "p1_private_counters": [],
            "p2_private_counters": [],
            "shared_counters": [],
            "onion_dispenser_locations": [(0, 1), (5, 1)],
            "dish_dispenser_locations": [(3, 4), (5, 4)],
            "pot_locations": [(4, 2), (4, 3)],
            "serve_locations": [(3, 1), (8, 1)],
        },
        'coordination_ring': {
            "counter_location_to_id": {
                (0, 1): 1,
                (1, 0): 2,
                (2, 0): 3,
                (4, 2): 4,
                (4, 3): 5,
                (3, 4): 6,
                (2, 2): 7,

            },
            "p1_private_counters": [],
            "p2_private_counters": [],
            "shared_counters": [],
            "onion_dispenser_locations": [(0, 3), (1, 4)],
            "dish_dispenser_locations": [(0, 2)],
            "pot_locations": [(3, 0), (4, 1)],
            "serve_locations": [(2, 4)],
        },
        'cramped_room': {
            "counter_location_to_id": {
                (0, 2): 1,
                (1, 0): 2,
                (3, 0): 3,
                (4, 2): 4,
                (2, 3): 5,

            },
            "p1_private_counters": [],
            "p2_private_counters": [],
            "shared_counters": [],
            "onion_dispenser_locations": [(0, 1), (4, 1)],
            "dish_dispenser_locations": [(1, 3)],
            "pot_locations": [(2, 0)],
            "serve_locations": [(3, 3)],
        },
    }
    return layout_params_dict[layout_name]


def process_no_pad(hidden_seqs):
    max_len = max(len(elem) for elem in hidden_seqs)
    X = []
    for i in range(len(hidden_seqs)):
        team_hs = hidden_seqs[i]
        team_hs_pad = team_hs
        # team_hs_mode = max(set(team_hs), key=team_hs.count)
        # for j in range(max_len):
        #     if j < len(team_hs):
        #         team_hs_pad.append(team_hs[j])
        #     else:
        #         team_hs_pad.append(team_hs_mode)
        X.append(team_hs_pad)
    # print('X', X)
    # return np.array(X)
    return X


def get_trial_data(data_dict, idx):
    participant_data = json_eval(data_dict[idx])['data']
    trial_num_to_data = {}
    current_agent = 0
    maxlen = len(participant_data)
    # maxlen = 10
    for i in range(1, maxlen):
        data = participant_data[i]
        #     print(data['trialdata']['layout_name'])
        #     break
        if 'trialdata' not in data or 'layout_name' not in data['trialdata']:
            #         print(data['trialdata'])
            continue
        current_layout = data['trialdata']['layout_name']
        if current_layout in ['training0', 'training2']:
            continue

        if current_layout not in trial_num_to_data:
            current_agent = 0
            trial_num_to_data[current_layout] = {}
            trial_num_to_data[current_layout][current_agent] = {}
            trial_num_to_data[current_layout][current_agent]['score'] = 0
            trial_num_to_data[current_layout][current_agent]['state_data'] = []
            trial_num_to_data[current_layout][current_agent]['joint_action'] = []
            trial_num_to_data[current_layout][current_agent]['time_elapsed'] = []

        time_left = data['trialdata']['time_left']
        if time_left == -1:
            #         print(data['trialdata'])
            trial_num_to_data[current_layout][current_agent]['score'] = data['trialdata']['score']
            trial_num_to_data[current_layout][current_agent]['state_data'].append(data['trialdata'])
            trial_num_to_data[current_layout][current_agent]['joint_action'].append(data['trialdata']['joint_action'])
            trial_num_to_data[current_layout][current_agent]['time_elapsed'].append(data['trialdata']['time_elapsed'])

            if current_agent == 0:
                current_agent = 1
                trial_num_to_data[current_layout][current_agent] = {}
                trial_num_to_data[current_layout][current_agent]['score'] = 0
                trial_num_to_data[current_layout][current_agent]['state_data'] = []
                trial_num_to_data[current_layout][current_agent]['joint_action'] = []
                trial_num_to_data[current_layout][current_agent]['time_elapsed'] = []
        else:
            trial_num_to_data[current_layout][current_agent]['state_data'].append(data['trialdata'])
            trial_num_to_data[current_layout][current_agent]['joint_action'].append(data['trialdata']['joint_action'])
            trial_num_to_data[current_layout][current_agent]['time_elapsed'].append(data['trialdata']['time_elapsed'])

    return trial_num_to_data

    # data[1]


def strategy_identification_fc(layout_name, layout_trials):

    layout_params = get_layout_params(layout_name)
    num_states_list = [2, 3, 4, 5, 6, 7]
    num_clusters_list = [2, 3, 4]
    # num_states_list = [5]
    # num_clusters_list = [2]

    arr = np.zeros((max(num_states_list) + 3, max(num_states_list) + 3))

    max_ss_score = 0
    best_combo = (0, 0)  # (n_states, n_clusters)
    best_assignment = []
    for n_states in num_states_list:
        test_unsuper_hmm, hidden_seqs, team_numbers, team_num_to_seq_probs = run_naive_hmm_on_p2(layout_trials, layout_name,
                                                                                                 layout_params,
                                                                                                 n_states=n_states)
        for n_clusters in num_clusters_list:
            print('hidden_seqs', hidden_seqs)
            cluster_labels, cluster_centers, ss = cluster_hidden_states(hidden_seqs, n_clusters=n_clusters)
            print(f'\nN={n_states}, K={n_clusters}: cluster_labels = {cluster_labels}, teams = {team_numbers}')
            print(f'N={n_states}, K={n_clusters}, sil. score = ', ss)
            arr[n_states, n_clusters] = ss
            if ss > max_ss_score:
                max_ss_score = ss
                best_combo = (n_states, n_clusters)
                best_assignment = cluster_labels

    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    plt.title("Layout: " + layout_name)
    plt.ylabel("Number of Hidden States")
    plt.xlabel("Number of Clusters")
    plt.xlim(num_clusters_list[0] - 0.5, num_clusters_list[-1] + 0.5)
    plt.ylim(num_states_list[0] - 0.5, num_states_list[-1] + 0.5)
    plt.savefig('testing2_' + layout_name + '_ss_score_12_7_21_2.png')
    plt.close()

    print("best combo", best_combo)
    print("best_assignment", best_assignment)

def get_pgms_data(layout_name):
    # Opening JSON file
    with open('../pgms_data/overcooked_pgms_gamedata.json') as f:

        # returns JSON object as
        # a dictionary
        data = json.load(f)

    data_dict = {}
    for i in range(len(data)):
        ind_data = data[i]['datastring']
        data_dict[i] = ind_data

    # dict_keys(['condition', 'counterbalance',
    # 'assignmentId', 'workerId', 'hitId', 'currenttrial',
    # 'bonus', 'data', 'questiondata', 'eventdata', 'useragent', 'mode', 'status'])

    all_trials = {}
    for trial_idx in data_dict:
        #     data = data_dict[trial_idx]
        trial_num_to_data = get_trial_data(data_dict, trial_idx)
        if 'coordination_ring' not in trial_num_to_data:
            continue
        all_trials[trial_idx] = trial_num_to_data

    layout_trials = {}
    for trial_idx in all_trials:
        agent_0_data = all_trials[trial_idx][layout_name][0]
        agent_1_data = all_trials[trial_idx][layout_name][1]
        trial_name0 = "trial_"+ str(trial_idx)+ "-agent_"+str(0)
        trial_name1 = "trial_" + str(trial_idx) + "-agent_" + str(1)
        layout_trials[trial_name0] = agent_0_data
        layout_trials[trial_name1] = agent_1_data
    return layout_trials


def compare_with_ppo():
    layout_trials = get_pgms_data(layout_name)
    g_dict = {}
    for key in layout_trials:


        ppo_path = '2021_11_30-06_40_42_STRATEXP_TEST1_random0_s1_weights_ppo_bc_train'
        agent_ppo, ppo_config = get_ppo_agent(ppo_path, 9456, best=True)
        # agent = get_agent_from_saved_model(save_dir + "/best", config["sim_threads"])


        simple_mdp = OvercookedGridworld.from_layout_name('random1', start_order_list=['any'], cook_time=20)
        base_params_start_or = {
            'start_orientations': True,
            'wait_allowed': False,
            'counter_goals': simple_mdp.terrain_pos_dict['X'],
            'counter_drop': [],
            'counter_pickup': [],
            'same_motion_goals': False
        }
        # mlp = MediumLevelPlanner(simple_mdp, base_params_start_or)
        agent_ppo.mdp = simple_mdp
        agent_ppo.agent_index = 0
        gamma = 0.999

        human_idx = 1
        total_g = 0
        for t in range(len(layout_trials[key]['state_data'])):
            inv_t = len(layout_trials[key]['state_data'])-t
            sample_state = layout_trials[key]['state_data'][t]['state']

            true_human_action = tuple(layout_trials[key]['joint_action'][t][human_idx])
            if true_human_action == ('I', 'N', 'T', 'E', 'R', 'A', 'C', 'T'):
                true_human_action = 'interact'
            # print("true_human_action", true_human_action)
            true_human_action_idx = Action.ACTION_TO_INDEX[true_human_action]
            # print("sample_state", sample_state)
            try:
                action_probs = agent_ppo.action(OvercookedState.from_dict(sample_state))
                print("action_probs", action_probs)
                print("true_human_action_idx", true_human_action_idx)

                robot_policy_of_true_human_action = action_probs[true_human_action_idx]
                robot_policy_of_predicted_human_action = max(action_probs)
                difference = abs(robot_policy_of_predicted_human_action-robot_policy_of_true_human_action)
                total_g += math.pow(gamma, inv_t) * difference
            except:
                continue

        g_dict[key] = total_g
    print("g_dict", g_dict)
    return g_dict

if __name__ == '__main__':
    layout_name = 'random0'
    # layout_trials = get_pgms_data(layout_name)
    # strategy_identification_fc(layout_name, layout_trials)
    compare_with_ppo()



