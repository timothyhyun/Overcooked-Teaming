from dependencies import *
from hmm import supervised_HMM, unsupervised_HMM, HiddenMarkovModel
from extract_features import *

################################### ACTIONS LIST ########################################
# 1. Pick up onion from dispenser
# 2. Pick up dish from dispenser
# 3. Put down onion on middle counter
# 4. Put down dish on middle counter
# 5. Put onion in top pot
# 6. Put onion in right pot
# 7. Get soup in dish from top pot
# 8. Get soup in dish from right pot
# 9. Serve soup
# 10. Put onion on private counter
# 11. Put dish on private counter
# 12. Put soup on private counter
# 13. Transporting onion
# 14. Transporting dish
# 15. Transporting soup
# 16. Moving without carrying
# 17. Doing nothing
# 18. Pick up onion from middle counter
# 19. Pick up dish from middle counter
# 20. Pick up onion from private counter
# 21. Pick up dish from private counter
# 22. Pick up soup from private counter
# 23. Pick up soup from middle counter
# 24. Put down soup on middle counter
# 25. Pad
#######################################################################################


def track_player_actions_first_attempt(old_trials, p1_data, p2_data, objects_data, p1_actions,
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

    # p1_major_action = 17  # initialize as doing nothing (stationary)
    # p2_major_action = 17
    # loop over your images
    for a in range(len(t) - 1):
        p1_major_action = 17  # initialize as doing nothing (stationary)
        p2_major_action = 17

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
                if p1_obj_name_next == 'onion':
                    if put_down_loc in p1_private_counters:
                        p1_major_action = 10
                    if put_down_loc in shared_counters:
                        p1_major_action = 3


                elif p1_obj_name_next == 'dish':
                    if put_down_loc in p1_private_counters:
                        p1_major_action = 11
                    if put_down_loc in shared_counters:
                        p1_major_action = 4

                elif p1_obj_name_next == 'soup':
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




            else:

                # placed at top counter pot
                if (placed_obj_x, placed_obj_y) == (3, 0):
                    top_soup_contents_dict['this_contents'].append(object_held_id)
                    right_soup_contents_dict['other_contents'].append(object_held_id)

                    if p2_obj_name == 'onion':
                        p2_major_action = 5

                # placed at right counter pot
                if (placed_obj_x, placed_obj_y) == (4, 1):
                    right_soup_contents_dict['this_contents'].append(object_held_id)
                    top_soup_contents_dict['other_contents'].append(object_held_id)

                    if p2_obj_name == 'onion':
                        p2_major_action = 6

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

        p1_actions_list.append((t1, p1_major_action))
        p2_actions_list.append((t1, p2_major_action))

    return object_list_tracker, ordered_delivered_tracker, p1_actions_list, p2_actions_list

def track_player_actions(old_trials, p1_data, p2_data, objects_data, p1_actions,
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

    # p1_major_action = 17  # initialize as doing nothing (stationary)
    # p2_major_action = 17
    # loop over your images
    for a in range(len(t) - 1):
        p1_major_action = 17  # initialize as doing nothing (stationary)
        p2_major_action = 17

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
                if p1_obj_name_next == 'onion':
                    if put_down_loc in p1_private_counters:
                        p1_major_action = 10
                    if put_down_loc in shared_counters:
                        p1_major_action = 3


                elif p1_obj_name_next == 'dish':
                    if put_down_loc in p1_private_counters:
                        p1_major_action = 11
                    if put_down_loc in shared_counters:
                        p1_major_action = 4

                elif p1_obj_name_next == 'soup':
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




            else:

                # placed at top counter pot
                if (placed_obj_x, placed_obj_y) == (3, 0):
                    top_soup_contents_dict['this_contents'].append(object_held_id)
                    right_soup_contents_dict['other_contents'].append(object_held_id)

                    if p2_obj_name == 'onion':
                        p2_major_action = 5

                # placed at right counter pot
                if (placed_obj_x, placed_obj_y) == (4, 1):
                    right_soup_contents_dict['this_contents'].append(object_held_id)
                    top_soup_contents_dict['other_contents'].append(object_held_id)

                    if p2_obj_name == 'onion':
                        p2_major_action = 6

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

        if p1_major_action == 17 and p2_major_action == 17:
            continue

        p1_actions_list.append((t1, p1_major_action))
        p2_actions_list.append((t1, p2_major_action))

    return object_list_tracker, ordered_delivered_tracker, p1_actions_list, p2_actions_list

def track_player_5_main_actions(old_trials, p1_data, p2_data, objects_data, p1_actions,
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
                if p1_obj_name_next == 'onion':
                    if put_down_loc in p1_private_counters:
                        p1_major_action = 10
                    if put_down_loc in shared_counters:
                        p1_major_action = 3


                elif p1_obj_name_next == 'dish':
                    if put_down_loc in p1_private_counters:
                        p1_major_action = 11
                    if put_down_loc in shared_counters:
                        p1_major_action = 4

                elif p1_obj_name_next == 'soup':
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




            else:

                # placed at top counter pot
                if (placed_obj_x, placed_obj_y) == (3, 0):
                    top_soup_contents_dict['this_contents'].append(object_held_id)
                    right_soup_contents_dict['other_contents'].append(object_held_id)

                    if p2_obj_name == 'onion':
                        p2_major_action = 5

                # placed at right counter pot
                if (placed_obj_x, placed_obj_y) == (4, 1):
                    right_soup_contents_dict['this_contents'].append(object_held_id)
                    top_soup_contents_dict['other_contents'].append(object_held_id)

                    if p2_obj_name == 'onion':
                        p2_major_action = 6

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

        p1_actions_list.append((t1, p1_major_action))
        p2_actions_list.append((t1, p2_major_action))

    return object_list_tracker, ordered_delivered_tracker, p1_actions_list, p2_actions_list

def featurize_data_for_granular_hmm():
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

        object_list_tracker, ordered_delivered_tracker, p1_actions_list, p2_actions_list = track_player_actions(old_trials,
                                                                                                                p1_data,
                                                                                                                p2_data,
                                                                                                                objects_data,
                                                                                                                p1_actions,
                                                                                                                p2_actions,
                                                                                                                name,
                                                                                                                time_elapsed)
        # print('p1_actions_list', p1_actions_list)
        # print('p2_actions_list', p2_actions_list)
        team_chunked_actions_data[trial_id] = {}
        team_chunked_actions_data[trial_id]['p1_actions'] = p1_actions_list
        team_chunked_actions_data[trial_id]['p2_actions'] = p2_actions_list

        team_numbers.append(trial_id)
    #     other_pot_contains_num_onions, other_pot_states, steps_p1_took_total_order, \
    #         steps_p2_took_total_order, order_completion_times = pull_features_from_output(object_list_tracker, ordered_delivered_tracker)

    #     plot_results(other_pot_contains_num_onions, other_pot_states, steps_p1_took_total_order, steps_p2_took_total_order, order_completion_times)
    #     print()
    return team_chunked_actions_data, team_numbers

def plot_actions_over_time(team_chunked_actions_data):
    for trial_id in team_chunked_actions_data:
        plt.figure(figsize=(10, 10))
        p1_actions_list = [elem[1] for elem in team_chunked_actions_data[trial_id]['p1_actions']]
        p2_actions_list = [elem[1] for elem in team_chunked_actions_data[trial_id]['p2_actions']]
        action_times_list = [elem[0] for elem in team_chunked_actions_data[trial_id]['p1_actions']]
        for i in range(len(action_times_list)):
            p1_act = p1_actions_list[i]
            p2_act = p2_actions_list[i]

            #         if p2_act == 9:
            #             color = 'r'
            #         else:
            color = 'g'
            plt.scatter(action_times_list[i], p1_actions_list[i], c=color)

        plt.title("Team: " + str(trial_id))
        plt.xlabel("Action Index")
        plt.ylabel("Second (time)")
        plt.show()
        plt.close()


def recalibrate_featurized_data(team_chunked_actions_data):
    X_data = []
    for trial_id in team_chunked_actions_data:
        data_add = [elem[1] for elem in team_chunked_actions_data[trial_id]['p2_actions']]
        # print('data_add', len(data_add))
        X_data.append(np.array(data_add))
    # X_data = np.array(X_data)

    unique_values = np.unique(X_data)
    n_unique_values = len(unique_values)

    old_action_val_to_new = {}
    for i in range(len(unique_values)):
        old_action_val_to_new[unique_values[i]] = i

    X_data_new = []
    for i in range(X_data.shape[0]):
        add = []
        for elem in X_data[i]:
            # print('elem', elem)
            add.append(old_action_val_to_new[elem])
        X_data_new.append(np.array(add))

    # X_data = np.array(X_data_new)
    X_data = X_data_new

    return X_data

def run_granular_hmm():

    # X = observation_data
    # Y = hidden_state_data
    team_chunked_actions_data, team_numbers = featurize_data_for_granular_hmm()
    X = recalibrate_featurized_data(team_chunked_actions_data)

    n_states = 2
    N_iters = 200

    test_unsuper_hmm = unsupervised_HMM(X, n_states, N_iters)

    # print('emission', test_unsuper_hmm.generate_emission(10))
    for j in range(len(X)):
        print('viterbi: hidden seq: Team ' + str(team_numbers[j]) + ": ", test_unsuper_hmm.viterbi(X[j]))

    return test_unsuper_hmm


if __name__ == '__main__':
    run_granular_hmm()




