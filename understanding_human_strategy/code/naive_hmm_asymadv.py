from dependencies import *
from hmm import supervised_HMM, unsupervised_HMM, HiddenMarkovModel
from extract_features import *
import sklearn
def track_player_actions_SIMPLE(old_trials, p1_data, p2_data, objects_data, p1_actions,
                                p2_actions, name, time_elapsed):
    # Save player 1 and 2 actions
    p1_actions_list = []
    p2_actions_list = []

    # HMM Actions
    hmm_observations = []

    N_steps = len(p1_data)
    a_min = 1  # the minimial value of the paramater a
    a_max = N_steps - 1  # the maximal value of the paramater a
    a_init = 1  # the value of the parameter a to be used initially, when the graph is created

    t = np.linspace(0, N_steps - 1, N_steps)

    # Define Counters for Counter Circuit
    counter_location_to_id = {
        (0,3): 1,
        (0,2): 2,
        (1,0): 3,
        (2,1): 4,
        (1,4): 5,
        (2,4): 6,
        (6,1): 7,
        (7,0): 8,
        (8,2): 9,
        (8,3): 10,
        (7,4): 11,
        (6,4): 12

    }
    counter_id_to_location = {v: k for k, v in counter_location_to_id.items()}

    # middle_counters = [13, 14, 15, 16]
    # middle_counter_locations = [counter_id_to_location[x] for x in middle_counters]

    all_counter_locations = list(counter_location_to_id.keys())

    # #     p1_private_counters = [4,5]
    # #     p2_private_counters = [6,7]
    # #     shared_counters = [1,2,3]
    #     p1_private_counters = [(1,0),(1,4)]
    #     p2_private_counters = [(4,2),(4,3)]
    #     shared_counters = [(2,1),(2,2),(2,3)]

    onion_dispenser_locations = [(0,1), (5,1)]
    dish_dispenser_locations = [(3,4), (5,4)]
    serve_locations = [(3,1), (8,1)]

    obj_count_id = 0
    next_obj_count_id = 0

    object_list_tracker = {}
    object_location_tracker = {}

    ordered_delivered_tracker = {}

    left_soup_counter_id = 0  # (3,0)
    right_soup_counter_id = 0  # (4,0)

    left_pot_loc = [4,2] #left = top
    right_pot_loc = [4,3] # right = bottom

    left_soup_contents_dict = {'this_contents': [], 'other_contents': [], 'other_state': 'empty'}  # (3,0)
    right_soup_contents_dict = {'this_contents': [], 'other_contents': [], 'other_state': 'empty'}  # (4,0)
    # states: empty, cooking, cooked, partial

    absolute_order_counter = 0

    p1_carrying_soup = None
    p1_carrying_soup_pot_side = None
    p1_carrying_soup_pot_side_id = None
    p1_time_picked_up_soup = None
    p1_time_delivered_soup = None

    p2_carrying_soup = None
    p2_carrying_soup_pot_side = None
    p2_carrying_soup_pot_side_id = None
    p2_time_picked_up_soup = None
    p2_time_delivered_soup = None

    players_holding = {1: None, 2: None}

    layout = eval(old_trials[old_trials['layout_name'] == name]['layout'].to_numpy()[0])
    layout = np.array([list(elem) for elem in layout])
    grid_display = np.zeros((layout.shape[0], layout.shape[1], 3))

    p1_major_action = -1  # initialize as doing nothing (null value)
    p2_major_action = -1
    # loop over your images
    for a in range(len(t) - 1):
        record_action_bool = False
        action_taken = None

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

        ################## PRINT STATE ##################
        # print("\n\nWORLD STATE")
        # print("players_holding", players_holding)
        # print("p1_data: ", p1_data[a])
        # print("p2_data: ", p2_data[a])
        # print("p1_actions: ", p1_actions[a])
        # print("p2_actions: ", p2_actions[a])
        # print("objects_data: ", objects_data[a])
        # print("\nNEXT STEP")
        # print("p1_data: ", p1_data[b])
        # print("p2_data: ", p2_data[b])
        # print("p1_actions: ", p1_actions[b])
        # print("p2_actions: ", p2_actions[b])
        # print("objects_data: ", objects_data[b])
        # print()

        ################## BEGIN TRACKING PLAYER 1'S MOVEMENT ##################
        # If P1 moves or stays
        if p1_act in ['N', 'S', 'E', 'W']:
            # A1: If P1 is carrying something and moving (dish or onion)
            if players_holding[1] is not None:
                obj_location = (p1_obj_x_next, p1_obj_y_next)
                object_held_id = players_holding[1]

                prev_location = object_list_tracker[object_held_id]['location']
                object_list_tracker[object_held_id]['n_actions_since_pickup'] += 1
                object_list_tracker[object_held_id]['location'] = obj_location
                object_list_tracker[object_held_id]['p1_n_actions_since_pickup'] += 1

                object_location_tracker.pop(prev_location, None)
                object_location_tracker[obj_location] = object_held_id

        # If P1 interacted
        if p1_act == 'I' and p1_actions[a]=='INTERACT':

            # If P1 picked up soup from the pot. P1 would be carrying a dish
            if p1_obj_name_next == 'soup':
                placed_x, placed_y = list(p1_x_next)[0] + p1_dir_x_next, list(p1_y_next)[0] + p1_dir_y_next
                # A2: If P1 filled up a dish with the soup
                if p1_obj_name == 'dish':
                    objects_status = objects_data[a]
                    objects_status_next = objects_data[b]
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
                    if (placed_x, placed_y) == tuple(right_pot_loc):
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
                if p1_obj_name_next == 'soup':
                    continue

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
                        hmm_observations.append(0)

                else:

                    # A5: P1 picked up an onion or dish from the counters
                    if picked_up_loc in object_location_tracker:
                        obj_picked_id = object_location_tracker[picked_up_loc]
                    else:
                        print('!!! problem p1 pickup not found')
                        nearest_key = min(list(object_location_tracker.keys()),
                                          key=lambda c: (c[0] - placed_x) ** 2 + (c[1] - placed_y) ** 2)
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

                    # If picked-up locations is in the middle counters
            #                     if p1_obj_name_next in ['onion', 'dish'] and picked_up_loc in middle_counter_locations:
            #                         if 1 in object_list_tracker[obj_picked_id]['player_holding_list'] and 2 in object_list_tracker[obj_picked_id]['player_holding_list']:
            #                             if object_list_tracker[obj_picked_id]['transport_method'] == 'middle_pass':
            #                                 record_action_bool = True
            #                                 action_taken = 0

            # If P1 put down an object
            if p1_obj_x is not None and p1_obj_x_next is None:
                object_held_id = players_holding[1]
                # print("..........................................")
                # print(".......................P1 object_held_id", object_held_id)
                # print("..........................................")
                placed_obj_x, placed_obj_y = list(p1_x_next)[0] + p1_dir_x_next, list(p1_y_next)[0] + p1_dir_y_next
                placed_location = (placed_obj_x, placed_obj_y)

                # A6: If P1 put an onion or dish on the counters
                if p1_obj_name in ['onion', 'dish'] and placed_location in all_counter_locations:
                    object_list_tracker[object_held_id]['player_holding'] = 0
                    object_list_tracker[object_held_id]['n_actions_since_pickup'] += 1
                    object_list_tracker[object_held_id]['player_holding_list'].append(0)
                    object_list_tracker[object_held_id]['p1_n_actions_since_pickup'] += 1

                    old_obj_location = object_list_tracker[object_held_id]['location']
                    new_obj_location = (placed_obj_x, placed_obj_y)
                    object_list_tracker[object_held_id]['location'] = new_obj_location

                    object_location_tracker.pop(old_obj_location, None)
                    object_location_tracker[new_obj_location] = object_held_id

                    counter_index = counter_location_to_id[(placed_obj_x, placed_obj_y)]
                    object_list_tracker[object_held_id]['counter_used'].append(counter_index)

                    # If the object was placed on a middle counter
                    # if counter_index in middle_counters:
                    #     object_list_tracker[object_held_id]['transport_method'] = 'middle_pass'

                    object_list_tracker[object_held_id]['p1_time_completed'].append(t2)

                    players_holding[1] = None
                    put_down_loc = new_obj_location


                # A7: If P1 delivered a soup
                elif p1_obj_name == 'soup' and placed_location in serve_locations:
                    p1_time_delivered_soup = t2
                    ordered_delivered_tracker[absolute_order_counter] = {}
                    ordered_delivered_tracker[absolute_order_counter]['details'] = p1_carrying_soup
                    ordered_delivered_tracker[absolute_order_counter]['pot_side'] = p1_carrying_soup_pot_side
                    ordered_delivered_tracker[absolute_order_counter]['pot_side_id'] = p1_carrying_soup_pot_side_id
                    ordered_delivered_tracker[absolute_order_counter]['time_picked_up'] = p1_time_picked_up_soup
                    ordered_delivered_tracker[absolute_order_counter]['time_delivered'] = p1_time_delivered_soup

                    absolute_order_counter += 1
                    players_holding[1] = None # Comment out maybe?

                    hmm_observations.append(1)


                # Else, A8: P1 placed an onion in one of the pots
                elif p1_obj_name == 'onion':

                    # print(f" A8: P1 placed an onion in one of the pots, object_held_id: {object_held_id}")

                    # placed at left counter pot
                    if placed_location == tuple(left_pot_loc):
                        left_soup_contents_dict['this_contents'].append(object_held_id)
                        right_soup_contents_dict['other_contents'].append(object_held_id)


                    # placed at right counter pot
                    if placed_location == tuple(right_pot_loc):
                        right_soup_contents_dict['this_contents'].append(object_held_id)
                        left_soup_contents_dict['other_contents'].append(object_held_id)


                    object_list_tracker[object_held_id]['player_holding'] = 0
                    object_list_tracker[object_held_id]['n_actions_since_pickup'] += 1
                    object_list_tracker[object_held_id]['player_holding_list'].append(0)
                    object_list_tracker[object_held_id]['p1_n_actions_since_pickup'] += 1

                    old_obj_location = object_list_tracker[object_held_id]['location']
                    new_obj_location = (placed_obj_x, placed_obj_y)
                    object_list_tracker[object_held_id]['location'] = new_obj_location

                    object_location_tracker.pop(old_obj_location, None)
                    object_location_tracker[new_obj_location] = object_held_id

                    if (placed_obj_x, placed_obj_y) in counter_location_to_id:
                        counter_index = counter_location_to_id[(placed_obj_x, placed_obj_y)]
                        object_list_tracker[object_held_id]['counter_used'].append(counter_index)
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

                    players_holding[1] = None

                # else:


        ################## END OF PLAYER 1'S MOVEMENT ##################

        ################## BEGIN TRACKNG PLAYER 2'S MOVEMENT ##################
        # If P2 moves or stays
        if p2_act in ['N', 'S', 'E', 'W']:
            # A1: If P2 is carrying something and moving (dish or onion)
            if players_holding[2] is not None:
                obj_location = (p2_obj_x_next, p2_obj_y_next)
                object_held_id = players_holding[2]

                prev_location = object_list_tracker[object_held_id]['location']
                object_list_tracker[object_held_id]['n_actions_since_pickup'] += 1
                object_list_tracker[object_held_id]['location'] = obj_location
                object_list_tracker[object_held_id]['p1_n_actions_since_pickup'] += 1

                object_location_tracker.pop(prev_location, None)
                object_location_tracker[obj_location] = object_held_id

        # If P2 interacted
        if p2_act == 'I' and p2_actions[a]=='INTERACT':

            # If P2 picked up soup from the pot. P2 would be carrying a dish
            if p2_obj_name_next == 'soup':
                placed_x, placed_y = list(p2_x_next)[0] + p2_dir_x_next, list(p2_y_next)[0] + p2_dir_y_next
                # A2: If P2 filled up a dish with the soup
                if p2_obj_name == 'dish':
                    objects_status = objects_data[a]
                    objects_status_next = objects_data[b]
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

                    p2_time_picked_up_soup = t1

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
                    if (placed_x, placed_y) == tuple(right_pot_loc):
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
                if p2_obj_name_next == 'soup':
                    continue

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
                    object_list_tracker[obj_count_id]['name'] = p1_obj_name_next
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
                        hmm_observations.append(2)


                else:

                    # A5: P2 picked up an onion or dish from the counters
                    if picked_up_loc in object_location_tracker:
                        obj_picked_id = object_location_tracker[picked_up_loc]
                    else:
                        # print('!!! problem p2 pickup not found')
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

                    object_location_tracker.pop(picked_up_loc, None)
                    object_location_tracker[new_obj_location] = obj_picked_id

                    # If picked-up locations is in the middle counters
            #                     if p1_obj_name_next in ['onion', 'dish'] and picked_up_loc in middle_counter_locations:
            #                         if 1 in object_list_tracker[obj_picked_id]['player_holding_list'] and 2 in object_list_tracker[obj_picked_id]['player_holding_list']:
            #                             if object_list_tracker[obj_picked_id]['transport_method'] == 'middle_pass':
            #                                 record_action_bool = True
            #                                 action_taken = 0

            # If P2 put down an object
            if p2_obj_x is not None and p2_obj_x_next is None:
                object_held_id = players_holding[2]
                placed_obj_x, placed_obj_y = list(p2_x_next)[0] + p2_dir_x_next, list(p2_y_next)[0] + p2_dir_y_next
                placed_location = (placed_obj_x, placed_obj_y)

                # A6: If P2 put an onion or dish on the counters
                if p2_obj_name in ['onion', 'dish'] and placed_location in all_counter_locations:
                    object_list_tracker[object_held_id]['player_holding'] = 0
                    object_list_tracker[object_held_id]['n_actions_since_pickup'] += 1
                    object_list_tracker[object_held_id]['player_holding_list'].append(0)
                    object_list_tracker[object_held_id]['p2_n_actions_since_pickup'] += 1

                    old_obj_location = object_list_tracker[object_held_id]['location']
                    new_obj_location = (placed_obj_x, placed_obj_y)
                    object_list_tracker[object_held_id]['location'] = new_obj_location

                    object_location_tracker.pop(old_obj_location, None)
                    object_location_tracker[new_obj_location] = object_held_id

                    counter_index = counter_location_to_id[(placed_obj_x, placed_obj_y)]
                    object_list_tracker[object_held_id]['counter_used'].append(counter_index)

                    # If the object was placed on a middle counter
                    # if counter_index in middle_counters:
                    #     object_list_tracker[object_held_id]['transport_method'] = 'middle_pass'

                    object_list_tracker[object_held_id]['p2_time_completed'].append(t2)

                    players_holding[2] = None
                    put_down_loc = new_obj_location


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

                    hmm_observations.append(3)

                # Else, A8: P2 placed an onion in one of the pots
                elif p2_obj_name == 'onion':

                    # placed at left counter pot
                    if placed_location == tuple(left_pot_loc):
                        left_soup_contents_dict['this_contents'].append(object_held_id)
                        right_soup_contents_dict['other_contents'].append(object_held_id)


                    # placed at right counter pot
                    if placed_location == tuple(right_pot_loc):
                        right_soup_contents_dict['this_contents'].append(object_held_id)
                        left_soup_contents_dict['other_contents'].append(object_held_id)


                    object_list_tracker[object_held_id]['player_holding'] = 0
                    object_list_tracker[object_held_id]['n_actions_since_pickup'] += 1
                    object_list_tracker[object_held_id]['player_holding_list'].append(0)
                    object_list_tracker[object_held_id]['p2_n_actions_since_pickup'] += 1

                    old_obj_location = object_list_tracker[object_held_id]['location']
                    new_obj_location = (placed_obj_x, placed_obj_y)
                    object_list_tracker[object_held_id]['location'] = new_obj_location

                    object_location_tracker.pop(old_obj_location, None)
                    object_location_tracker[new_obj_location] = object_held_id

                    if (placed_obj_x, placed_obj_y) in counter_location_to_id:
                        counter_index = counter_location_to_id[(placed_obj_x, placed_obj_y)]
                        object_list_tracker[object_held_id]['counter_used'].append(counter_index)
                        # if counter_index in middle_counters:
                        #     object_list_tracker[object_held_id]['transport_method'] = 'middle_pass'

                    object_list_tracker[object_held_id]['p2_time_completed'].append(t2)

                    players_holding[2] = None

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

        ##### SAVE TO LISTS
        # if record_action_bool == True:
        #     hmm_observations.append(action_taken)
    print(hmm_observations)

    return object_list_tracker, ordered_delivered_tracker, hmm_observations




def featurize_data_for_naive_hmm(window=4, ss=2):
    team_chunked_actions_data = {}

    name = 'asymmetric_advantages'
    title = 'Asymmetric Advantages'

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

        object_list_tracker, ordered_delivered_tracker, hmm_observations = track_player_actions_SIMPLE(old_trials, p1_data, p2_data,
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
    #
    # # X_data = np.array(X_data)
    #
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


def run_naive_hmm_on_p2(n_states):
    # X = observation_data
    # Y = hidden_state_data

    X, team_numbers = featurize_data_for_naive_hmm(window=window, ss=ss)


    N_iters = 100

    test_unsuper_hmm = unsupervised_HMM(X, n_states, N_iters)

    # print('emission', test_unsuper_hmm.generate_emission(10))
    hidden_seqs = []
    for j in range(len(X)):
        viterbi_output = test_unsuper_hmm.viterbi(X[j])
        hidden_seqs.append([int(x) for x in viterbi_output])
        print('viterbi: hidden seq: Team ' + str(team_numbers[j]) + ": ", viterbi_output)

    return test_unsuper_hmm, hidden_seqs, team_numbers

def run_naive_hmm_on_p2_method_2(n_states, window=4, ss=2):

    # X = observation_data
    # Y = hidden_state_data
    X, team_numbers, X_data_chunked, team_numbers_chunked = featurize_data_for_naive_hmm(window=window, ss=ss)


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

def cluster_hidden_states(hidden_seqs, n_clusters=2):
    X = pad_w_mode(hidden_seqs)
    # print('X=', X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    ss = sklearn.metrics.silhouette_score(X, cluster_labels)
    return cluster_labels, cluster_centers, ss


def cluster_hidden_states_agglo(hidden_seqs, n_clusters=2):
    from sklearn.cluster import AgglomerativeClustering
    # X = pad_w_mode(hidden_seqs)
    # print('X=', X)
    X = process_no_pad(hidden_seqs)
    acluster = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
    cluster_labels = acluster.labels_
    cluster_centers = acluster.cluster_centers_
    return cluster_labels, cluster_centers
#
#
# if __name__ == '__main__':
#     test_unsuper_hmm, hidden_seqs, team_numbers = run_naive_hmm_on_p2()
#
#     # Try N=2 Clusters
#     cluster_labels, cluster_centers = cluster_hidden_states_agglo(hidden_seqs, n_clusters=2)
#     print(f'\nN=2: cluster_labels = {cluster_labels}, teams = {team_numbers}')
#
#     # Try N=3 Clusters
#     # cluster_labels, cluster_centers = cluster_hidden_states_agglo(hidden_seqs, n_clusters=3)
#     # print(f'\nN=3: cluster_labels = {cluster_labels}, teams = {team_numbers}')
#     #
#     # # Try N=4 Clusters
#     # cluster_labels, cluster_centers = cluster_hidden_states_agglo(hidden_seqs, n_clusters=4)
#     # print(f'\nN=4: cluster_labels = {cluster_labels}, teams = {team_numbers}')
#     #
#     # # Try N=5 Clusters
#     # cluster_labels, cluster_centers = cluster_hidden_states_agglo(hidden_seqs, n_clusters=5)
#     # print(f'\nN=3: cluster_labels = {cluster_labels}, teams = {team_numbers}')
#
#
def plot_validation_matrix():
    # num_states_list = [2,3,4]
    # num_clusters_list = [2, 3, 4]
    num_states_list = [4]
    num_clusters_list = [2]

    arr = np.zeros((5, 5))

    for n_states in num_states_list:
        test_unsuper_hmm, hidden_seqs, team_numbers, team_num_to_seq_probs = run_naive_hmm_on_p2_method_2(n_states=n_states, window=9, ss=3)
        for n_clusters in num_clusters_list:



            cluster_labels, cluster_centers, ss = cluster_hidden_states(hidden_seqs, n_clusters=n_clusters)
            print(f'\nN={n_states}, K={n_clusters}: cluster_labels = {cluster_labels}, teams = {team_numbers}')
            print(f'N={n_states}, K={n_clusters}, sil. score = ', ss)
            arr[n_states, n_clusters] = ss



    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    plt.title("Asym Adv")
    plt.ylabel("Number of Hidden States")
    plt.xlabel("Number of Clusters")
    plt.xlim(num_clusters_list[0] - 0.5, num_clusters_list[-1] + 0.5)
    plt.ylim(num_states_list[0] - 0.5, num_states_list[-1] + 0.5)
    plt.savefig('aa_ch_score.png')
    plt.close()

# if __name__ == '__main__':
#     num_states_list = [2,3,4, 5, 6]
#     num_clusters_list = [2, 3, 4, 5, 6]
#
#
#
#
#     # num_states_list = [4]
#     # num_clusters_list = [2]
#
#     arr = np.zeros((7,7))
#
#     for n_states in num_states_list:
#         test_unsuper_hmm, hidden_seqs, team_numbers = run_naive_hmm_on_p2(n_states=n_states)
#         for n_clusters in num_clusters_list:
#             cluster_labels, cluster_centers, ss = cluster_hidden_states(hidden_seqs, n_clusters=n_clusters)
#             print(f'\nN={n_states}, K={n_clusters}: cluster_labels = {cluster_labels}, teams = {team_numbers}')
#             print(f'N={n_states}, K={n_clusters}, sil. score = ', ss)
#             arr[n_states, n_clusters] = ss
#
#     plt.imshow(arr, cmap='viridis')
#     plt.colorbar()
#     plt.title("Asym Adv")
#     plt.ylabel("Number of Hidden States")
#     plt.xlabel("Number of Clusters")
#     plt.savefig('aa_ss2.png')
#


if __name__ == '__main__':
    plot_validation_matrix()