import itertools
import numpy as np

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.planning.planners import Heuristic
from overcooked_ai_py.planning.search import SearchTree
from overcooked_ai_py.agents.agent import Agent, AgentGroup, CoupledPlanningAgent
from overcooked_ai_py.planning.fixed_strat_planners import LowLevel_MotionPlanner, MediumLevel_MotionPlanner

    
    
EMPTY = 0
PARTIALLY_FILLED = 1
COOKING = 2
READY = 3

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

NORTH = (0, -1)
SOUTH = (0, 1)
EAST  = (1, 0)
WEST  = (-1, 0)

ALL_DIRECTIONS = [NORTH, EAST, SOUTH, WEST]
#
# ORIENTATION_TO_FACING = {}
# FACING_TO_ORIENTATION = {}
# for i in range(len(ALL_DIRECTIONS)):
#     ORIENTATION_TO_FACING[]


class DualPotPolicy():
    def __init__(self, mdp, player_index):
        self.mdp = mdp
        # self.medium_level_planner = medium_level_planner
        self.player_index = player_index
        self.partner_index = 1-player_index
        self.reset_game()
        self.setup_planners()

    def setup_planners(self):
        # self.hl_action_manager = HighLevelActionManager(self.medium_level_planner)
        # self.hl_planner = HighLevelPlanner(self.hl_action_manager)
        self.med_level_planner = MediumLevel_MotionPlanner(self.mdp, self.player_index)
        

    def make_pot_tracker(self):
        self.pot_tracker = {}
        for location in self.pot_locations:
            self.pot_tracker[location] = {}
            self.pot_tracker[location]['contents'] = []
            self.pot_tracker[location]['state'] = EMPTY

    def make_free_objects_tracker(self):
        self.free_onion_tracker = []
        self.free_dish_tracker = []
        self.free_soup_tracker = []

    def make_players_tracker(self):
        self.player_tracker = {}
        for i in [0,1]:
            self.player_tracker[i] = {}
            self.player_tracker[i]['orientation'] = self.players[i].orientation
            self.player_tracker[i]['holding'] = self.players[i].held_object
            self.player_tracker[i]['position'] = self.players[i].position
            self.player_tracker[i]['goal_loc'] = None

    def update_trackers(self, objects, state):
        # Resolve new player locations
        # Update free objects tracking
        self.update_with_recent_joint_action_and_state(self.old_state)


        # Update player tracking
        for i in [0, 1]:
            self.player_tracker[i]['orientation'] = self.players[i].orientation
            self.player_tracker[i]['position'] = self.players[i].position
            self.player_tracker[i]['holding'] = self.players[i].held_object

        # Update pot tracking
        pot_states = self.mdp.get_pot_states(state)
        self.update_pot_states(state)
        # print('pot_states', pot_states)

    def update_pot_states(self, state):
        pot_states = self.mdp.get_pot_states(state)
        empty_pots = pot_states["empty"]
        ready_pots = pot_states["tomato"]["ready"] + pot_states["onion"]["ready"]
        cooking_pots = pot_states["tomato"]["cooking"] + pot_states["onion"]["cooking"]
        nearly_ready_pots = pot_states["tomato"]["partially_full"] + pot_states["onion"][
            "partially_full"]
        full_pots = cooking_pots + ready_pots
        num_pots = len(pot_states)

        # print('ready_pots', ready_pots)
        # print('cooking_pots', cooking_pots)
        # print('nearly_ready_pots', nearly_ready_pots)
        # print('empty_pots', empty_pots)


        # for loc in ready_pots:
        #     self.pot_tracker[loc]['state'] = READY
        # for loc in cooking_pots:
        #     self.pot_tracker[loc]['state'] = COOKING
        # for loc in nearly_ready_pots:
        #     self.pot_tracker[loc]['state'] = PARTIALLY_FILLED
        # for loc in empty_pots:
        #     self.pot_tracker[loc]['state'] = EMPTY
        # print('self.pottracker', self.pot_tracker)

        for pot_pos in self.mdp.get_pot_locations():
            if not state.has_object(pot_pos):
                self.pot_tracker[pot_pos]['contents'] = []
                self.pot_tracker[pot_pos]['state'] = EMPTY
            else:
                soup_obj = state.get_object(pot_pos)
                soup_type, num_items, cook_time = soup_obj.state
                if 0 < num_items < self.mdp.num_items_for_soup:
                    self.pot_tracker[pot_pos]['contents'] = ['onion' for _ in range(num_items)]
                    self.pot_tracker[pot_pos]['state'] = PARTIALLY_FILLED
                elif num_items == self.mdp.num_items_for_soup:
                    if cook_time >= self.mdp.soup_cooking_time:
                        self.pot_tracker[pot_pos]['contents'] = ['onion' for _ in range(num_items)]
                        self.pot_tracker[pot_pos]['state'] = READY
                    else:
                        self.pot_tracker[pot_pos]['contents'] = ['onion' for _ in range(num_items)]
                        self.pot_tracker[pot_pos]['state'] = COOKING
        # print('self.pottracker', self.pot_tracker)



    def reset_game(self):
        self.pot_locations = self.mdp.get_pot_locations()
        self.onion_dispenser_locations = self.mdp.get_onion_dispenser_locations()
        self.counter_locations = self.mdp.get_counter_locations()
        self.dish_dispenser_locations = self.mdp.get_dish_dispenser_locations()
        self.serving_locations = self.mdp.get_serving_locations()
        self.players = self.mdp.get_standard_start_state().players
        self.player_positions = self.mdp.start_player_positions

        self.make_free_objects_tracker()
        self.make_pot_tracker()
        self.make_players_tracker()

        self.recent_player_actions = ('stay', 'stay')
        self.old_state = self.mdp.get_standard_start_state()
        self.num_items_for_soup = 3

        self.current_action_sequence = []
        self.fill_mode = True

        self.recent_player_actions = (Action.STAY, Action.STAY)

        self.passing_counters = [(2,1), (2,2), (2,3)]

    def update_with_joint_action(self, joint_action):
        self.recent_player_actions = joint_action
        # print("self.recent_player_actions", self.recent_player_actions)

    def update_on_state(self, state):
        # print('state', state)
        self.players = state.players
        objects = state.objects
        self.update_trackers(objects, state)

        self.old_state = state


    def update_with_recent_joint_action_and_state(self, new_state):
        """
        Resolve any INTERACT actions, if present. Update trackers appropriately

        Currently if two players both interact with a terrain, we resolve player 1's interact
        first and then player 2's, without doing anything like collision checking.
        """
        joint_action = self.recent_player_actions

        pot_states = self.mdp.get_pot_states(new_state)

        ready_pots = pot_states["tomato"]["ready"] + pot_states["onion"]["ready"]
        cooking_pots = ready_pots + pot_states["tomato"]["cooking"] + pot_states["onion"]["cooking"]
        nearly_ready_pots = cooking_pots + pot_states["tomato"]["partially_full"] + pot_states["onion"][
            "partially_full"]
        full_pots = cooking_pots + ready_pots
        num_pots = len(pot_states)


        for pot_pos in self.mdp.get_pot_locations():
            if not new_state.has_object(pot_pos):
                self.pot_tracker[pot_pos]['contents'] = []
                self.pot_tracker[pot_pos]['state'] = EMPTY
            else:
                soup_obj = new_state.get_object(pot_pos)
                soup_type, num_items, cook_time = soup_obj.state
                if 0 < num_items < self.mdp.num_items_for_soup:
                    self.pot_tracker[pot_pos]['contents'] = ['onion' for _ in range(num_items)]
                    self.pot_tracker[pot_pos]['state'] = PARTIALLY_FILLED
                elif num_items == self.mdp.num_items_for_soup:
                    if cook_time >= self.mdp.soup_cooking_time:
                        self.pot_tracker[pot_pos]['contents'] = ['onion' for _ in range(num_items)]
                        self.pot_tracker[pot_pos]['state'] = READY
                    else:
                        self.pot_tracker[pot_pos]['contents'] = ['onion' for _ in range(num_items)]
                        self.pot_tracker[pot_pos]['state'] = COOKING

        counters_considered = self.mdp.terrain_pos_dict['X']
        self.free_onion_tracker = []
        self.free_dish_tracker = []
        self.free_soup_tracker = []
        for obj in new_state.objects.values():
            if obj.position in counters_considered and obj.position in self.passing_counters:
                # counter_objects_dict[obj.name].append(obj.position)
                if obj.name == 'onion':
                    self.free_onion_tracker.append(obj.position)
                if obj.name == 'dish':
                    self.free_dish_tracker.append(obj.position)
                if obj.name == 'soup':
                    self.free_soup_tracker.append(obj.position)

        # player_idx = -1
        # for player, action in zip(new_state.players, joint_action):
        #
        #     player_idx += 1
        #     print("UPDATING FOR FIXED PLAYER: ", player_idx)
        #
        #     # if action != Action.INTERACT:
        #     #     continue
        #
        #     pos, o = player.position, player.orientation
        #     i_pos = Action.move_in_direction(pos, o)
        #     terrain_type = self.mdp.get_terrain_type_at_pos(i_pos)
        #
        #     if terrain_type == 'X':
        #
        #         if player.has_object() and not new_state.has_object(i_pos):
        #             # Action Type 1: Player put object on counter
        #
        #             # If player placed an obj on counter
        #             # if player.get_object().name in ['onion', 'tomato', 'dish']:
        #             if player.get_object().name == "onion":
        #                 self.free_onion_tracker.append(i_pos)
        #             if player.get_object().name == "dish":
        #                 self.free_dish_tracker.append(i_pos)
        #             if player.get_object().name == "soup":
        #                 self.free_soup_tracker.append(i_pos)
        #
        #             # print('\n\n\n PUT DOWN')
        #             # print("onion tracker", self.free_onion_tracker)
        #             # print("dish tracker", self.free_dish_tracker)
        #             # print("soup tracker", self.free_soup_tracker)
        #
        #
        #         elif not player.has_object() and new_state.has_object(i_pos):
        #             # Action Type 2: Player picked object up from counter
        #
        #             # print('\n\n\n PICKUP')
        #             # print('new_state', new_state)
        #             # print("onion tracker", self.free_onion_tracker)
        #             # print("dish tracker", self.free_dish_tracker)
        #             # print("soup tracker", self.free_soup_tracker)
        #
        #             obj_name = new_state.get_object(i_pos).name
        #             if obj_name == 'onion':
        #                 self.free_onion_tracker.remove(i_pos)
        #             if obj_name == 'dish':
        #                 self.free_dish_tracker.remove(i_pos)
        #             if obj_name == 'soup':
        #                 self.free_soup_tracker.remove(i_pos)
        #
        #     elif terrain_type == 'O' and player.held_object is None:
        #         # Action Type 3: Player picked up onion from dispenser
        #         pass
        #
        #
        #     elif terrain_type == 'D' and player.held_object is None:
        #         # Action Type 5: Player picked up dish from dispenser
        #         pass
        #
        #     elif terrain_type == 'P' and player.has_object():
        #         if player.get_object().name == 'dish' and new_state.has_object(i_pos):
        #             # Action Type 6: Player picked up soup from pot with dish
        #             obj_name = new_state.get_object(i_pos).name
        #             self.pot_tracker[i_pos]['contents'] = []
        #             self.pot_tracker[i_pos]['state'] = EMPTY
        #
        #
        #
        #         elif player.get_object().name in ['onion', 'tomato']:
        #             item_type = player.get_object().name
        #
        #             if not new_state.has_object(i_pos):
        #                 # Action Type 7: Player placed onion or tomato in empty pot
        #                 # Pot was empty
        #                 obj_name = player.get_object().name
        #                 self.pot_tracker[i_pos]['contents'].append(obj_name)
        #                 self.pot_tracker[i_pos]['state'] = PARTIALLY_FILLED
        #
        #
        #             else:
        #                 # Action Type 8: Player placed onion in partially filled pot
        #                 # Pot has already items in it
        #                 obj = new_state.get_object(i_pos)
        #                 soup_type, num_items, cook_time = obj.state
        #
        #                 if num_items < self.num_items_for_soup and soup_type == item_type:
        #                     obj_name = player.get_object().name
        #                     self.pot_tracker[i_pos]['contents'].append(obj_name)
        #                     self.pot_tracker[i_pos]['state'] = PARTIALLY_FILLED
        #
        #                     if num_items + 1 == self.num_items_for_soup:
        #                         self.pot_tracker[i_pos]['state'] = COOKING
        #
        #
        #
        #     elif terrain_type == 'S' and player.has_object():
        #         obj = player.get_object()
        #         if obj.name == 'soup':
        #             # Action Type 9: Player delivered soup
        #             pass

        return


    def get_action_sequence(self, state):
        self.players = state.players
        for i in [0, 1]:
            self.player_tracker[i]['orientation'] = self.players[i].orientation
            self.player_tracker[i]['position'] = self.players[i].position
            self.player_tracker[i]['holding'] = self.players[i].held_object

        pot_states = self.mdp.get_pot_states(state)
        self.update_pot_states(state)

        self.update_with_recent_joint_action_and_state(state)
        # print("pot_states", pot_states)

        player_pos = self.player_tracker[self.player_index]['position']
        player_or = self.player_tracker[self.player_index]['orientation']

        counter_locations = [(2, 1), (2, 2), (2, 3)]
        # If player is server (P0)
        if self.player_index == 0:

            pot_loc_0 = self.pot_locations[0]
            pot_loc_1 = self.pot_locations[1]

            pot_loc_0_needs_onions = False
            pot_loc_1_needs_onions = False

            if len(self.pot_tracker[pot_loc_0]['contents']) < self.num_items_for_soup:
                pot_loc_0_needs_onions = True

            if len(self.pot_tracker[pot_loc_1]['contents']) < self.num_items_for_soup:
                pot_loc_1_needs_onions = True

            # print('pot_loc_0_needs_onions = ',pot_loc_0_needs_onions)
            # print("pot_loc_1_needs_onions = ", pot_loc_1_needs_onions)

            # Case 1. If player0 is carrying an soup
            if self.player_tracker[self.player_index]['holding'] is not None and self.player_tracker[self.player_index]['holding'].name == 'soup':
                action_sequence = self.med_level_planner.serve_soup(player_pos, player_or)
                self.current_action_sequence = action_sequence

            # Case 2. If player0 is carrying an onion
            elif self.player_tracker[self.player_index]['holding'] is not None and self.player_tracker[self.player_index]['holding'].name == 'onion':
                if pot_loc_0_needs_onions:
                    action_sequence = self.med_level_planner.put_onion_in_pot(player_pos, player_or, pot_loc_0)
                    self.current_action_sequence = action_sequence
                elif pot_loc_1_needs_onions:
                    action_sequence = self.med_level_planner.put_onion_in_pot(player_pos, player_or, pot_loc_1)
                    self.current_action_sequence = action_sequence

            # Case 3. If player0 is carrying a dish
            elif self.player_tracker[self.player_index]['holding'] is not None and self.player_tracker[self.player_index]['holding'].name == 'dish':
                if self.pot_tracker[pot_loc_0]['state'] == READY:
                    action_sequence = self.med_level_planner.pickup_soup_from_pot(player_pos, player_or, pot_loc_0)
                    self.current_action_sequence = action_sequence
                elif self.pot_tracker[pot_loc_1]['state'] == READY:
                    action_sequence = self.med_level_planner.pickup_soup_from_pot(player_pos, player_or, pot_loc_1)
                    self.current_action_sequence = action_sequence

            # Case 4. If player0 is holding nothing
            elif self.player_tracker[self.player_index]['holding']== None:
                # Case 4b. If both pots are ready, pick up dish
                # else:
                if self.fill_mode is False:
                    if pot_loc_0_needs_onions == False:
                        # If there are free dishes on the counter, pick them up
                        if len(self.free_dish_tracker) > 0:
                            free_dish_location = self.free_dish_tracker[0]
                            action_sequence = self.med_level_planner.pickup_object_from_passing_counter(player_pos, player_or, free_dish_location)
                            self.current_action_sequence = action_sequence

                    elif pot_loc_1_needs_onions == False:
                        # If there are free dishes on the counter, pick them up
                        if len(self.free_dish_tracker) > 0:
                            free_dish_location = self.free_dish_tracker[0]
                            action_sequence = self.med_level_planner.pickup_object_from_passing_counter(player_pos,
                                                                                                        player_or,
                                                                                                        free_dish_location)
                            self.current_action_sequence = action_sequence

                    else:
                        self.fill_mode = True

                # Case 4a. If there are empty or partially filled pots, pick up onion
                if self.fill_mode:
                    if pot_loc_1_needs_onions or pot_loc_0_needs_onions:
                        # print("POTS NEED ONIONS")
                        # If there are free onions on the counter, pick them up
                        if len(self.free_onion_tracker) > 0:
                            # print("EXIST FREE ONIONS")
                            # print("self.free_onion_tracker", self.free_onion_tracker)
                            free_onion_location = self.free_onion_tracker[0]
                            # if free_onion_location in counter_locations:
                            action_sequence = self.med_level_planner.pickup_object_from_passing_counter(player_pos, player_or, free_onion_location)
                            # print('action_sequence', action_sequence)
                            self.current_action_sequence = action_sequence

                    if pot_loc_1_needs_onions is False and pot_loc_0_needs_onions is False:
                        self.fill_mode = False

                # Case 4b. If both pots are ready, pick up dish
                # else:
                #     if pot_loc_0_needs_onions == False or pot_loc_1_needs_onions == False:
                #         # If there are free dishes on the counter, pick them up
                #         if len(self.free_dish_tracker) > 0:
                #             free_dish_location = self.free_dish_tracker[0]
                #             action_sequence = self.med_level_planner.pickup_object_from_passing_counter(player_pos, player_or, free_dish_location)
                #             self.current_action_sequence = action_sequence

        if self.player_index == 1:

            pot_loc_0 = self.pot_locations[0]
            pot_loc_1 = self.pot_locations[1]

            pot_loc_0_needs_onions = False
            pot_loc_1_needs_onions = False

            if len(self.pot_tracker[pot_loc_0]['contents']) < self.num_items_for_soup:
                pot_loc_0_needs_onions = True

            if len(self.pot_tracker[pot_loc_1]['contents']) < self.num_items_for_soup:
                pot_loc_1_needs_onions = True

            # print('pot_loc_0_needs_onions = ', pot_loc_0_needs_onions)
            # print("pot_loc_1_needs_onions = ", pot_loc_1_needs_onions)

            # Case 1. If player1 is carrying an onion
            if self.player_tracker[self.player_index]['holding'] is not None and self.player_tracker[self.player_index]['holding'].name == 'onion':
                # print("CARRYING ONION, PUT DOWN")
                action_sequence = self.med_level_planner.put_onion_on_passing_counter(player_pos, player_or)
                # print('action_sequence', action_sequence)
                self.current_action_sequence = action_sequence


            # Case 2. If player0 is carrying a dish
            if self.player_tracker[self.player_index]['holding'] is not None and self.player_tracker[self.player_index]['holding'].name == 'dish':
                action_sequence = self.med_level_planner.put_dish_on_passing_counter(player_pos, player_or)
                self.current_action_sequence = action_sequence


            # Case 4. If player1 is holding nothing
            elif self.player_tracker[self.player_index]['holding'] == None:
                if self.fill_mode is False:
                    if pot_loc_0_needs_onions == False and pot_loc_1_needs_onions == False:
                        # If there are free dishes on the counter, pick them up
                        p0_holding_dish = 0
                        if self.player_tracker[0]['holding'] is not None and self.player_tracker[0][
                            'holding'].name == 'dish':
                            p0_holding_dish = 1
                        if len(self.free_dish_tracker) + p0_holding_dish < 2:
                            action_sequence = self.med_level_planner.pickup_dish_from_dispenser(player_pos, player_or)
                            self.current_action_sequence = action_sequence

                    elif pot_loc_0_needs_onions == False:
                        # If there are free dishes on the counter, pick them up
                        p0_holding_dish = 0
                        if self.player_tracker[0]['holding'] is not None and self.player_tracker[0][
                            'holding'].name == 'dish':
                            p0_holding_dish = 1
                        if len(self.free_dish_tracker) + p0_holding_dish < 1:
                            action_sequence = self.med_level_planner.pickup_dish_from_dispenser(player_pos, player_or)
                            self.current_action_sequence = action_sequence

                    elif pot_loc_1_needs_onions == False:
                        # If there are free dishes on the counter, pick them up
                        p0_holding_dish = 0
                        if self.player_tracker[0]['holding'] is not None and self.player_tracker[0][
                            'holding'].name == 'dish':
                            p0_holding_dish = 1
                        if len(self.free_dish_tracker) + p0_holding_dish < 1:
                            action_sequence = self.med_level_planner.pickup_dish_from_dispenser(player_pos, player_or)
                            self.current_action_sequence = action_sequence

                    else:
                        self.fill_mode = True

                # Case 4a. If there are empty or partially filled pots, pick up onion
                if self.fill_mode:
                    if pot_loc_1_needs_onions or pot_loc_0_needs_onions:
                        # If there are enough free onions
                        p0_holding_onion = 0
                        if self.player_tracker[0]['holding'] is not None and self.player_tracker[0]['holding'].name == 'onion':
                            p0_holding_onion = 1
                        if (len(self.free_onion_tracker) + len(self.pot_tracker[pot_loc_0]['contents']) + len(self.pot_tracker[pot_loc_1]['contents'])) + p0_holding_onion >= 6:
                            action_sequence = []
                            self.current_action_sequence = action_sequence

                        # If there are enough free onions
                        elif len(self.free_onion_tracker) + p0_holding_onion > 0:
                            action_sequence = []
                            self.current_action_sequence = action_sequence

                        else:
                            # print("MUST PICKUP ONION")
                            action_sequence = self.med_level_planner.pickup_onion_from_dispenser(player_pos, player_or)
                            # print('action_sequence', action_sequence)
                            self.current_action_sequence = action_sequence

                    if pot_loc_0_needs_onions == False and pot_loc_1_needs_onions == False:
                        self.fill_mode = False

                # Case 4b. If both pots are ready, pick up dish
                # else:
                #     if pot_loc_0_needs_onions == False and pot_loc_1_needs_onions == False:
                #         # If there are free dishes on the counter, pick them up
                #         if len(self.free_dish_tracker) <= 1:
                #             action_sequence = self.med_level_planner.pickup_dish_from_dispenser(player_pos, player_or)
                #             self.current_action_sequence = action_sequence



    def get_action(self, state):
        # self.players = state.players
        # objects = state.objects
        # self.update_trackers(objects, state)
        # print(f"\n\n GET ACTION, PIndex: {self.player_index}")
        # print('self.player_tracker', self.player_tracker)
        # print("free onions", self.free_onion_tracker)
        # print("free dishes", self.free_dish_tracker)
        # print("pot tracking", self.pot_tracker)
        # print('self.current_action_sequence', self.current_action_sequence)
        # print("fill mode", self.fill_mode)
        
        if len(self.current_action_sequence) > 0:
            # selected_action = self.current_action_sequence[0]
            selected_action = self.current_action_sequence.pop(0)

        else:
            self.get_action_sequence(state)
            # print("self.current_action_sequence", self.current_action_sequence)
            if len(self.current_action_sequence) > 0:
                selected_action = self.current_action_sequence[0]
                self.current_action_sequence.pop(0)
            else:
                selected_action = Action.STAY

        # print(f"PIndex: {self.player_index}, ACTION: {selected_action}")
        return selected_action


class FixedStrategy_AgentPair(AgentGroup):
    """
    AgentPair is the N=2 case of AgentGroup. Unlike AgentGroup,
    it supports having both agents being the same instance of Agent.

    NOTE: Allowing duplicate agents (using the same instance of an agent
    for both fields can lead to problems if the agents have state / history)
    """

    def __init__(self, *agents, allow_duplicate_agents=False):
        super().__init__(*agents, allow_duplicate_agents=allow_duplicate_agents)
        assert self.n == 2
        self.a0, self.a1 = self.agents

        if type(self.a0) is CoupledPlanningAgent and type(self.a1) is CoupledPlanningAgent:
            print(
                "If the two planning agents have same params, consider using CoupledPlanningPair instead to reduce computation time by a factor of 2")

    def joint_action(self, state):
        if self.a0 is self.a1:
            # When using the same instance of an agent for self-play,
            # reset agent index at each turn to prevent overwriting it
            self.a0.set_agent_index(0)
            action_0 = self.a0.action(state)
            self.a1.set_agent_index(1)
            action_1 = self.a1.action(state)

            self.a0.update_with_joint_action((action_0, action_1))
            self.a1.update_with_joint_action((action_0, action_1))

            return (action_0, action_1)
        else:
            # print("UPDATE HERE")
            selected_joint_action = super().joint_action(state)
            self.a0.update_with_joint_action(selected_joint_action)
            self.a1.update_with_joint_action(selected_joint_action)
            return selected_joint_action



class DualPotAgent(Agent):
    """
    An agent that plays dual pot strategy.
    NOTE: Does not perform interact actions
    """

    def __init__(self, mdp, player_index, sim_threads=None):
        self.sim_threads = sim_threads
        self.mdp = mdp
        self.player_index = player_index
        # self.medium_level_planner = medium_level_planner

        self.action_policy = DualPotPolicy(self.mdp, self.player_index)
        self.action_policy.reset_game()

    def set_agent_index(self, agent_index):
        self.player_index = agent_index
        self.action_policy = DualPotPolicy(self.mdp, self.player_index)
        self.action_policy.reset_game()

    def read_state(self, state):
        print('state', state)

    def update_action_policy_on_state(self, state):
        self.action_policy.update_on_state(state)

    def update_with_joint_action(self, joint_action):
        self.action_policy.update_with_joint_action(joint_action)


    def action(self, state):
        self.update_action_policy_on_state(state)

        idx = np.random.randint(4)
        action_selected = self.action_policy.get_action(state)

        return action_selected

    def actions(self, states, agent_indices):
        actions = []
        for state, agent_idx in zip(states, agent_indices):
            self.set_agent_index(agent_idx)
            self.reset()
            actions.append(self.action(state))
        return actions

    def direct_action(self, obs):
        return [np.random.randint(4) for _ in range(self.sim_threads)]


class SinglePotPolicy():
    def __init__(self, mdp, player_index):
        self.mdp = mdp
        # self.medium_level_planner = medium_level_planner
        self.player_index = player_index
        self.partner_index = 1 - player_index
        self.reset_game()
        self.setup_planners()

    def setup_planners(self):
        # self.hl_action_manager = HighLevelActionManager(self.medium_level_planner)
        # self.hl_planner = HighLevelPlanner(self.hl_action_manager)
        self.med_level_planner = MediumLevel_MotionPlanner(self.mdp, self.player_index)

    def make_pot_tracker(self):
        self.pot_tracker = {}
        for location in self.pot_locations:
            self.pot_tracker[location] = {}
            self.pot_tracker[location]['contents'] = []
            self.pot_tracker[location]['state'] = EMPTY

    def make_free_objects_tracker(self):
        self.free_onion_tracker = []
        self.free_dish_tracker = []
        self.free_soup_tracker = []

    def make_players_tracker(self):
        self.player_tracker = {}
        for i in [0, 1]:
            self.player_tracker[i] = {}
            self.player_tracker[i]['orientation'] = self.players[i].orientation
            self.player_tracker[i]['holding'] = self.players[i].held_object
            self.player_tracker[i]['position'] = self.players[i].position
            self.player_tracker[i]['goal_loc'] = None

    def update_trackers(self, objects, state):
        # Resolve new player locations
        # Update free objects tracking
        self.update_with_recent_joint_action_and_state(self.old_state)

        # Update player tracking
        for i in [0, 1]:
            self.player_tracker[i]['orientation'] = self.players[i].orientation
            self.player_tracker[i]['position'] = self.players[i].position
            self.player_tracker[i]['holding'] = self.players[i].held_object

        # Update pot tracking
        pot_states = self.mdp.get_pot_states(state)
        self.update_pot_states(state)
        # print('pot_states', pot_states)

    def update_pot_states(self, state):
        pot_states = self.mdp.get_pot_states(state)
        empty_pots = pot_states["empty"]
        ready_pots = pot_states["tomato"]["ready"] + pot_states["onion"]["ready"]
        cooking_pots = pot_states["tomato"]["cooking"] + pot_states["onion"]["cooking"]
        nearly_ready_pots = pot_states["tomato"]["partially_full"] + pot_states["onion"][
            "partially_full"]
        full_pots = cooking_pots + ready_pots
        num_pots = len(pot_states)

        # print('ready_pots', ready_pots)
        # print('cooking_pots', cooking_pots)
        # print('nearly_ready_pots', nearly_ready_pots)
        # print('empty_pots', empty_pots)

        # for loc in ready_pots:
        #     self.pot_tracker[loc]['state'] = READY
        # for loc in cooking_pots:
        #     self.pot_tracker[loc]['state'] = COOKING
        # for loc in nearly_ready_pots:
        #     self.pot_tracker[loc]['state'] = PARTIALLY_FILLED
        # for loc in empty_pots:
        #     self.pot_tracker[loc]['state'] = EMPTY
        # print('self.pottracker', self.pot_tracker)
        for pot_pos in self.mdp.get_pot_locations():
            if not state.has_object(pot_pos):
                self.pot_tracker[pot_pos]['contents'] = []
                self.pot_tracker[pot_pos]['state'] = EMPTY
            else:
                soup_obj = state.get_object(pot_pos)
                soup_type, num_items, cook_time = soup_obj.state
                if 0 < num_items < self.mdp.num_items_for_soup:
                    self.pot_tracker[pot_pos]['contents'] = ['onion' for _ in range(num_items)]
                    self.pot_tracker[pot_pos]['state'] = PARTIALLY_FILLED
                elif num_items == self.mdp.num_items_for_soup:
                    if cook_time >= self.mdp.soup_cooking_time:
                        self.pot_tracker[pot_pos]['contents'] = ['onion' for _ in range(num_items)]
                        self.pot_tracker[pot_pos]['state'] = READY
                    else:
                        self.pot_tracker[pot_pos]['contents'] = ['onion' for _ in range(num_items)]
                        self.pot_tracker[pot_pos]['state'] = COOKING

        # print('self.pottracker', self.pot_tracker)

    def reset_game(self):
        self.pot_locations = self.mdp.get_pot_locations()
        self.onion_dispenser_locations = self.mdp.get_onion_dispenser_locations()
        self.counter_locations = self.mdp.get_counter_locations()
        self.dish_dispenser_locations = self.mdp.get_dish_dispenser_locations()
        self.serving_locations = self.mdp.get_serving_locations()
        self.players = self.mdp.get_standard_start_state().players
        self.player_positions = self.mdp.start_player_positions

        self.make_free_objects_tracker()
        self.make_pot_tracker()
        self.make_players_tracker()

        self.recent_player_actions = ('stay', 'stay')
        self.old_state = self.mdp.get_standard_start_state()
        self.num_items_for_soup = 3

        self.current_action_sequence = []
        self.fill_mode = True

        self.passing_counters = [(2, 1), (2, 2), (2, 3)]

    def update_with_joint_action(self, joint_action):
        self.recent_player_actions = joint_action
        # print("self.recent_player_actions", self.recent_player_actions)

    def update_on_state(self, state):
        # print('state', state)
        self.players = state.players
        objects = state.objects
        self.update_trackers(objects, state)

        self.old_state = state

    def update_with_recent_joint_action_and_state(self, new_state):
        """
        Resolve any INTERACT actions, if present. Update trackers appropriately

        Currently if two players both interact with a terrain, we resolve player 1's interact
        first and then player 2's, without doing anything like collision checking.
        """
        joint_action = self.recent_player_actions

        pot_states = self.mdp.get_pot_states(new_state)

        ready_pots = pot_states["tomato"]["ready"] + pot_states["onion"]["ready"]
        cooking_pots = ready_pots + pot_states["tomato"]["cooking"] + pot_states["onion"]["cooking"]
        nearly_ready_pots = cooking_pots + pot_states["tomato"]["partially_full"] + pot_states["onion"][
            "partially_full"]
        full_pots = cooking_pots + ready_pots
        num_pots = len(pot_states)

        for pot_pos in self.mdp.get_pot_locations():
            if not new_state.has_object(pot_pos):
                self.pot_tracker[pot_pos]['contents'] = []
                self.pot_tracker[pot_pos]['state'] = EMPTY
            else:
                soup_obj = new_state.get_object(pot_pos)
                soup_type, num_items, cook_time = soup_obj.state
                if 0 < num_items < self.mdp.num_items_for_soup:
                    self.pot_tracker[pot_pos]['contents'] = ['onion' for _ in range(num_items)]
                    self.pot_tracker[pot_pos]['state'] = PARTIALLY_FILLED
                elif num_items == self.mdp.num_items_for_soup:
                    if cook_time >= self.mdp.soup_cooking_time:
                        self.pot_tracker[pot_pos]['contents'] = ['onion' for _ in range(num_items)]
                        self.pot_tracker[pot_pos]['state'] = READY
                    else:
                        self.pot_tracker[pot_pos]['contents'] = ['onion' for _ in range(num_items)]
                        self.pot_tracker[pot_pos]['state'] = COOKING

        counters_considered = self.mdp.terrain_pos_dict['X']
        self.free_onion_tracker = []
        self.free_dish_tracker = []
        self.free_soup_tracker = []
        for obj in new_state.objects.values():
            if obj.position in counters_considered and obj.position in self.passing_counters:
                # counter_objects_dict[obj.name].append(obj.position)
                if obj.name == 'onion':
                    self.free_onion_tracker.append(obj.position)
                if obj.name == 'dish':
                    self.free_dish_tracker.append(obj.position)
                if obj.name == 'soup':
                    self.free_soup_tracker.append(obj.position)

        # player_idx = -1
        # for player, action in zip(new_state.players, joint_action):
        #     player_idx += 1
        #
        #     print("UPDATING FOR FIXED PLAYER: ", player_idx)
        #     # if action != Action.INTERACT:
        #     #     continue
        #
        #     pos, o = player.position, player.orientation
        #     i_pos = Action.move_in_direction(pos, o)
        #     terrain_type = self.mdp.get_terrain_type_at_pos(i_pos)
        #
        #     if terrain_type == 'X':
        #
        #         if player.has_object() and not new_state.has_object(i_pos):
        #             # Action Type 1: Player put object on counter
        #
        #             # If player placed an obj on counter
        #             # if player.get_object().name in ['onion', 'tomato', 'dish']:
        #             if player.get_object().name == "onion":
        #                 self.free_onion_tracker.append(i_pos)
        #             if player.get_object().name == "dish":
        #                 self.free_dish_tracker.append(i_pos)
        #             if player.get_object().name == "soup":
        #                 self.free_soup_tracker.append(i_pos)
        #
        #             # print('\n\n\n PUT DOWN')
        #             # print("onion tracker", self.free_onion_tracker)
        #             # print("dish tracker", self.free_dish_tracker)
        #             # print("soup tracker", self.free_soup_tracker)
        #
        #
        #         elif not player.has_object() and new_state.has_object(i_pos):
        #             # Action Type 2: Player picked object up from counter
        #
        #             # print('\n\n\n PICKUP')
        #             # print('new_state', new_state)
        #             # print("onion tracker", self.free_onion_tracker)
        #             # print("dish tracker", self.free_dish_tracker)
        #             # print("soup tracker", self.free_soup_tracker)
        #
        #             obj_name = new_state.get_object(i_pos).name
        #             if obj_name == 'onion':
        #                 self.free_onion_tracker.remove(i_pos)
        #             if obj_name == 'dish':
        #                 self.free_dish_tracker.remove(i_pos)
        #             if obj_name == 'soup':
        #                 self.free_soup_tracker.remove(i_pos)
        #
        #     elif terrain_type == 'O' and player.held_object is None:
        #         # Action Type 3: Player picked up onion from dispenser
        #         pass
        #
        #
        #     elif terrain_type == 'D' and player.held_object is None:
        #         # Action Type 5: Player picked up dish from dispenser
        #         pass
        #
        #     elif terrain_type == 'P' and player.has_object():
        #         if player.get_object().name == 'dish' and new_state.has_object(i_pos):
        #             # Action Type 6: Player picked up soup from pot with dish
        #             obj_name = new_state.get_object(i_pos).name
        #             self.pot_tracker[i_pos]['contents'] = []
        #             self.pot_tracker[i_pos]['state'] = EMPTY
        #
        #
        #
        #         elif player.get_object().name in ['onion', 'tomato']:
        #             item_type = player.get_object().name
        #
        #             if not new_state.has_object(i_pos):
        #                 # Action Type 7: Player placed onion or tomato in empty pot
        #                 # Pot was empty
        #                 obj_name = player.get_object().name
        #                 self.pot_tracker[i_pos]['contents'].append(obj_name)
        #                 self.pot_tracker[i_pos]['state'] = PARTIALLY_FILLED
        #
        #
        #             else:
        #                 # Action Type 8: Player placed onion in partially filled pot
        #                 # Pot has already items in it
        #                 obj = new_state.get_object(i_pos)
        #                 soup_type, num_items, cook_time = obj.state
        #
        #                 if num_items < self.num_items_for_soup and soup_type == item_type:
        #                     obj_name = player.get_object().name
        #                     self.pot_tracker[i_pos]['contents'].append(obj_name)
        #                     self.pot_tracker[i_pos]['state'] = PARTIALLY_FILLED
        #
        #                     if num_items + 1 == self.num_items_for_soup:
        #                         self.pot_tracker[i_pos]['state'] = COOKING
        #
        #
        #
        #     elif terrain_type == 'S' and player.has_object():
        #         obj = player.get_object()
        #         if obj.name == 'soup':
        #             # Action Type 9: Player delivered soup
        #             pass

        return

    def get_action_sequence(self, state):
        self.players = state.players
        for i in [0, 1]:
            self.player_tracker[i]['orientation'] = self.players[i].orientation
            self.player_tracker[i]['position'] = self.players[i].position
            self.player_tracker[i]['holding'] = self.players[i].held_object

        pot_states = self.mdp.get_pot_states(state)
        self.update_pot_states(state)
        self.update_with_recent_joint_action_and_state(state)

        # print("pot_states", pot_states)

        player_pos = self.player_tracker[self.player_index]['position']
        player_or = self.player_tracker[self.player_index]['orientation']

        counter_locations = [(2, 1), (2, 2), (2, 3)]
        # If player is server (P0)
        if self.player_index == 0:

            pot_loc_0 = self.pot_locations[0]
            pot_loc_1 = self.pot_locations[1]

            pot_loc_0_needs_onions = False
            pot_loc_1_needs_onions = False

            if len(self.pot_tracker[pot_loc_0]['contents']) < self.num_items_for_soup:
                pot_loc_0_needs_onions = True

            # if pot_loc_0_needs_onions:
            #     self.fill_mode = True
            # else:
            #     self.fill_mode = False

            if len(self.pot_tracker[pot_loc_1]['contents']) < self.num_items_for_soup:
                pot_loc_1_needs_onions = True

            # print('pot_loc_0_needs_onions = ', pot_loc_0_needs_onions)
            # print("pot_loc_1_needs_onions = ", pot_loc_1_needs_onions)

            # Case 1. If player0 is carrying an soup
            if self.player_tracker[self.player_index]['holding'] is not None and self.player_tracker[self.player_index][
                'holding'].name == 'soup':
                action_sequence = self.med_level_planner.serve_soup(player_pos, player_or)
                self.current_action_sequence = action_sequence

            # Case 2. If player0 is carrying an onion
            elif self.player_tracker[self.player_index]['holding'] is not None and \
                    self.player_tracker[self.player_index]['holding'].name == 'onion':
                if pot_loc_0_needs_onions:
                    action_sequence = self.med_level_planner.put_onion_in_pot(player_pos, player_or, pot_loc_0)
                    self.current_action_sequence = action_sequence
                # elif pot_loc_1_needs_onions:
                #     action_sequence = self.med_level_planner.put_onion_in_pot(player_pos, player_or, pot_loc_1)
                #     self.current_action_sequence = action_sequence

            # Case 3. If player0 is carrying a dish
            elif self.player_tracker[self.player_index]['holding'] is not None and \
                    self.player_tracker[self.player_index]['holding'].name == 'dish':
                if self.pot_tracker[pot_loc_0]['state'] == READY:
                    action_sequence = self.med_level_planner.pickup_soup_from_pot(player_pos, player_or, pot_loc_0)
                    self.current_action_sequence = action_sequence
                # elif self.pot_tracker[pot_loc_1]['state'] == READY:
                #     action_sequence = self.med_level_planner.pickup_soup_from_pot(player_pos, player_or, pot_loc_1)
                #     self.current_action_sequence = action_sequence

            # Case 4. If player0 is holding nothing
            elif self.player_tracker[self.player_index]['holding'] == None:
                # Case 4b. If both pots are ready, pick up dish
                # else:
                if self.fill_mode is False:
                    if pot_loc_0_needs_onions == False:
                        # If there are free dishes on the counter, pick them up
                        if len(self.free_dish_tracker) > 0:
                            free_dish_location = self.free_dish_tracker[0]
                            action_sequence = self.med_level_planner.pickup_object_from_passing_counter(player_pos,
                                                                                                        player_or,
                                                                                                        free_dish_location)
                            self.current_action_sequence = action_sequence

                    # elif pot_loc_1_needs_onions == False:
                    #     # If there are free dishes on the counter, pick them up
                    #     if len(self.free_dish_tracker) > 0:
                    #         free_dish_location = self.free_dish_tracker[0]
                    #         action_sequence = self.med_level_planner.pickup_object_from_passing_counter(player_pos,
                    #                                                                                     player_or,
                    #                                                                                     free_dish_location)
                    #         self.current_action_sequence = action_sequence

                    # if pot_loc_0_needs_onions == True and pot_loc_1_needs_onions == True:
                    else:
                        self.fill_mode = True

                # Case 4a. If there are empty or partially filled pots, pick up onion
                if self.fill_mode:
                    if pot_loc_0_needs_onions:
                        # print("POTS NEED ONIONS")
                        # If there are free onions on the counter, pick them up
                        if len(self.free_onion_tracker) > 0:
                            # print("EXIST FREE ONIONS")
                            # print("self.free_onion_tracker", self.free_onion_tracker)
                            free_onion_location = self.free_onion_tracker[0]
                            # if free_onion_location in counter_locations:
                            action_sequence = self.med_level_planner.pickup_object_from_passing_counter(player_pos,
                                                                                                        player_or,
                                                                                                        free_onion_location)
                            # print('action_sequence', action_sequence)
                            self.current_action_sequence = action_sequence
                    else:
                        self.fill_mode = False

                # Case 4b. If both pots are ready, pick up dish
                # else:
                #     if pot_loc_0_needs_onions == False or pot_loc_1_needs_onions == False:
                #         # If there are free dishes on the counter, pick them up
                #         if len(self.free_dish_tracker) > 0:
                #             free_dish_location = self.free_dish_tracker[0]
                #             action_sequence = self.med_level_planner.pickup_object_from_passing_counter(player_pos, player_or, free_dish_location)
                #             self.current_action_sequence = action_sequence

        if self.player_index == 1:

            pot_loc_0 = self.pot_locations[0]
            pot_loc_1 = self.pot_locations[1]

            pot_loc_0_needs_onions = False
            pot_loc_1_needs_onions = False

            if len(self.pot_tracker[pot_loc_0]['contents']) < self.num_items_for_soup:
                pot_loc_0_needs_onions = True

            if len(self.pot_tracker[pot_loc_1]['contents']) < self.num_items_for_soup:
                pot_loc_1_needs_onions = True

            # print('pot_loc_0_needs_onions = ', pot_loc_0_needs_onions)
            # print("pot_loc_1_needs_onions = ", pot_loc_1_needs_onions)

            # Case 1. If player1 is carrying an onion
            if self.player_tracker[self.player_index]['holding'] is not None and self.player_tracker[self.player_index][
                'holding'].name == 'onion':
                # print("CARRYING ONION, PUT DOWN")
                action_sequence = self.med_level_planner.put_onion_on_passing_counter(player_pos, player_or)
                # print('action_sequence', action_sequence)
                self.current_action_sequence = action_sequence

            # Case 2. If player0 is carrying a dish
            if self.player_tracker[self.player_index]['holding'] is not None and self.player_tracker[self.player_index][
                'holding'].name == 'dish':
                action_sequence = self.med_level_planner.put_dish_on_passing_counter(player_pos, player_or)
                self.current_action_sequence = action_sequence


            # Case 4. If player1 is holding nothing
            elif self.player_tracker[self.player_index]['holding'] == None:
                if self.fill_mode is False:
                    if pot_loc_0_needs_onions == False:
                        # If there are no free dishes on the counter, pick up a dish
                        p0_holding_dish = 0
                        if self.player_tracker[0]['holding'] is not None and self.player_tracker[0][
                            'holding'].name == 'dish':
                            p0_holding_dish = 1

                        if len(self.free_dish_tracker) + p0_holding_dish < 1:
                            action_sequence = self.med_level_planner.pickup_dish_from_dispenser(player_pos, player_or)
                            self.current_action_sequence = action_sequence

                    # elif pot_loc_1_needs_onions == False:
                    #     # If there are free dishes on the counter, pick them up
                    #     if len(self.free_dish_tracker) < 1:
                    #         action_sequence = self.med_level_planner.pickup_dish_from_dispenser(player_pos, player_or)
                    #         self.current_action_sequence = action_sequence

                    if pot_loc_0_needs_onions == True:
                        self.fill_mode = True

                # Case 4a. If there are empty or partially filled pots, pick up onion
                if self.fill_mode:
                    if pot_loc_0_needs_onions:
                        # If there are enough free onions
                        p0_holding_onion = 0
                        if self.player_tracker[0]['holding'] is not None and self.player_tracker[0]['holding'].name == 'onion':
                            p0_holding_onion = 1
                        if (len(self.free_onion_tracker) + len(self.pot_tracker[pot_loc_0]['contents'])) + p0_holding_onion >= 3:
                            action_sequence = []
                            self.current_action_sequence = action_sequence

                        # If there are not enough free onions
                        elif len(self.free_onion_tracker) + p0_holding_onion > 0:
                            action_sequence = []
                            self.current_action_sequence = action_sequence

                        else:
                            # print("MUST PICKUP ONION")
                            action_sequence = self.med_level_planner.pickup_onion_from_dispenser(player_pos, player_or)
                            # print('action_sequence', action_sequence)
                            self.current_action_sequence = action_sequence

                    else:
                        self.fill_mode = False

                # Case 4b. If both pots are ready, pick up dish
                # else:
                #     if pot_loc_0_needs_onions == False and pot_loc_1_needs_onions == False:
                #         # If there are free dishes on the counter, pick them up
                #         if len(self.free_dish_tracker) <= 1:
                #             action_sequence = self.med_level_planner.pickup_dish_from_dispenser(player_pos, player_or)
                #             self.current_action_sequence = action_sequence

    def get_action(self, state):
        # self.players = state.players
        # objects = state.objects
        # self.update_trackers(objects, state)
        # print(f"\n\n GET ACTION, PIndex: {self.player_index}")
        # print('self.player_tracker', self.player_tracker)
        # print("free onions", self.free_onion_tracker)
        # print("free dishes", self.free_dish_tracker)
        # print("pot tracking", self.pot_tracker)
        # print('self.current_action_sequence', self.current_action_sequence)
        # print("fill mode", self.fill_mode)

        if len(self.current_action_sequence) > 0:
            # selected_action = self.current_action_sequence[0]
            selected_action = self.current_action_sequence.pop(0)

        else:
            self.get_action_sequence(state)
            # print("self.current_action_sequence", self.current_action_sequence)
            if len(self.current_action_sequence) > 0:
                selected_action = self.current_action_sequence[0]
                self.current_action_sequence.pop(0)
            else:
                selected_action = Action.STAY

        # print(f"PIndex: {self.player_index}, ACTION: {selected_action}")
        return selected_action


class SinglePotAgent(Agent):
    """
    An agent that plays single pot strategy.
    NOTE: Does not perform interact actions
    """

    def __init__(self, mdp, player_index, sim_threads=None):
        self.sim_threads = sim_threads
        self.mdp = mdp
        self.player_index = player_index
        # self.medium_level_planner = medium_level_planner

        self.action_policy = SinglePotPolicy(self.mdp, self.player_index)
        self.action_policy.reset_game()

    def set_agent_index(self, agent_index):
        self.player_index = agent_index
        self.action_policy = SinglePotPolicy(self.mdp, self.player_index)
        self.action_policy.reset_game()

    def read_state(self, state):
        print('state', state)

    def update_action_policy_on_state(self, state):
        self.action_policy.update_on_state(state)

    def update_with_joint_action(self, joint_action):
        self.action_policy.update_with_joint_action(joint_action)


    def action(self, state):
        self.update_action_policy_on_state(state)

        idx = np.random.randint(4)
        action_selected = self.action_policy.get_action(state)

        return action_selected

    def actions(self, states, agent_indices):
        actions = []
        for state, agent_idx in zip(states, agent_indices):
            self.set_agent_index(agent_idx)
            self.reset()
            actions.append(self.action(state))
        return actions

    def direct_action(self, obs):
        return [np.random.randint(4) for _ in range(self.sim_threads)]




class HighLevelActionManager(object):
    """
    Manager for high level actions. Determines available high level actions
    for each state and player.
    """

    def __init__(self, medium_level_planner):
        self.mdp = medium_level_planner.mdp

        self.wait_allowed = medium_level_planner.params['wait_allowed']
        self.counter_drop = medium_level_planner.params["counter_drop"]
        self.counter_pickup = medium_level_planner.params["counter_pickup"]

        self.mlp = medium_level_planner
        self.ml_action_manager = medium_level_planner.ml_action_manager
        self.mp = medium_level_planner.mp

    def joint_hl_actions(self, state):
        hl_actions_a0, hl_actions_a1 = tuple(self.get_high_level_actions(state, player) for player in state.players)
        joint_hl_actions = list(itertools.product(hl_actions_a0, hl_actions_a1))

        assert self.mlp.params["same_motion_goals"]
        valid_joint_hl_actions = joint_hl_actions

        if len(valid_joint_hl_actions) == 0:
            print("WARNING: found a state without high level successors")
        return valid_joint_hl_actions

    def get_high_level_actions(self, state, player):
        player_hl_actions = []
        counter_pickup_objects = self.mdp.get_counter_objects_dict(state, self.counter_pickup)
        if player.has_object():
            place_obj_ml_actions = self.ml_action_manager.get_medium_level_actions(state, player)

            # HACK to prevent some states not having successors due to lack of waiting actions
            if len(place_obj_ml_actions) == 0:
                place_obj_ml_actions = self.ml_action_manager.get_medium_level_actions(state, player,
                                                                                       waiting_substitute=True)

            place_obj_hl_actions = [HighLevelAction([ml_action]) for ml_action in place_obj_ml_actions]
            player_hl_actions.extend(place_obj_hl_actions)
        else:
            pot_states_dict = self.mdp.get_pot_states(state)
            player_hl_actions.extend(self.get_onion_and_put_in_pot(state, counter_pickup_objects, pot_states_dict))
            player_hl_actions.extend(self.get_tomato_and_put_in_pot(state, counter_pickup_objects, pot_states_dict))
            player_hl_actions.extend(self.get_dish_and_soup_and_serve(state, counter_pickup_objects, pot_states_dict))
        return player_hl_actions

    def get_dish_and_soup_and_serve(self, state, counter_objects, pot_states_dict):
        """Get all sequences of medium-level actions (hl actions) that involve a player getting a dish,
        going to a pot and picking up a soup, and delivering the soup."""
        dish_pickup_actions = self.ml_action_manager.pickup_dish_actions(state, counter_objects)
        pickup_soup_actions = self.ml_action_manager.pickup_soup_with_dish_actions(pot_states_dict)
        deliver_soup_actions = self.ml_action_manager.deliver_soup_actions()
        hl_level_actions = list(itertools.product(dish_pickup_actions, pickup_soup_actions, deliver_soup_actions))
        return [HighLevelAction(hl_action_list) for hl_action_list in hl_level_actions]

    def get_onion_and_put_in_pot(self, state, counter_objects, pot_states_dict):
        """Get all sequences of medium-level actions (hl actions) that involve a player getting an onion
        from a dispenser and placing it in a pot."""
        onion_pickup_actions = self.ml_action_manager.pickup_onion_actions(state, counter_objects)
        put_in_pot_actions = self.ml_action_manager.put_onion_in_pot_actions(pot_states_dict)
        hl_level_actions = list(itertools.product(onion_pickup_actions, put_in_pot_actions))
        return [HighLevelAction(hl_action_list) for hl_action_list in hl_level_actions]

    def get_tomato_and_put_in_pot(self, state, counter_objects, pot_states_dict):
        """Get all sequences of medium-level actions (hl actions) that involve a player getting an tomato
        from a dispenser and placing it in a pot."""
        tomato_pickup_actions = self.ml_action_manager.pickup_tomato_actions(state, counter_objects)
        put_in_pot_actions = self.ml_action_manager.put_tomato_in_pot_actions(pot_states_dict)
        hl_level_actions = list(itertools.product(tomato_pickup_actions, put_in_pot_actions))
        return [HighLevelAction(hl_action_list) for hl_action_list in hl_level_actions]


class HighLevelPlanner(object):
    """A planner that computes optimal plans for two agents to
    deliver a certain number of dishes in an OvercookedGridworld
    using high level actions in the corresponding A* search problems
    """

    def __init__(self, hl_action_manager):
        self.hl_action_manager = hl_action_manager
        self.mlp = self.hl_action_manager.mlp
        self.jmp = self.mlp.ml_action_manager.joint_motion_planner
        self.mp = self.jmp.motion_planner
        self.mdp = self.mlp.mdp

    def get_successor_states(self, start_state):
        """Determines successor states for high-level actions"""
        successor_states = []

        if self.mdp.is_terminal(start_state):
            return successor_states

        for joint_hl_action in self.hl_action_manager.joint_hl_actions(start_state):
            _, end_state, hl_action_cost = self.perform_hl_action(joint_hl_action, start_state)

            successor_states.append((joint_hl_action, end_state, hl_action_cost))
        return successor_states

    def perform_hl_action(self, joint_hl_action, curr_state):
        """Determines the end state for a high level action, and the corresponding low level action plan and cost.
        Will return Nones if a pot exploded throughout the execution of the action"""
        full_plan = []
        motion_goal_indices = (0, 0)
        total_cost = 0
        while not self.at_least_one_finished_hl_action(joint_hl_action, motion_goal_indices):
            curr_jm_goal = tuple(joint_hl_action[i].motion_goals[motion_goal_indices[i]] for i in range(2))
            joint_motion_action_plans, end_pos_and_ors, plan_costs = \
                self.jmp.get_low_level_action_plan(curr_state.players_pos_and_or, curr_jm_goal)
            curr_state = self.jmp.derive_state(curr_state, end_pos_and_ors, joint_motion_action_plans)
            motion_goal_indices = self._advance_motion_goal_indices(motion_goal_indices, plan_costs)
            total_cost += min(plan_costs)
            full_plan.extend(joint_motion_action_plans)
        return full_plan, curr_state, total_cost

    def at_least_one_finished_hl_action(self, joint_hl_action, motion_goal_indices):
        """Returns whether either agent has reached the end of the motion goal list it was supposed
        to perform to finish it's high level action"""
        return any([len(joint_hl_action[i].motion_goals) == motion_goal_indices[i] for i in range(2)])

    def get_low_level_action_plan(self, start_state, h_fn, debug=False):
        """
        Get a plan of joint-actions executable in the environment that will lead to a goal number of deliveries
        by performaing an A* search in high-level action space

        Args:
            state (OvercookedState): starting state

        Returns:
            full_joint_action_plan (list): joint actions to reach goal
            cost (int): a cost in number of timesteps to reach the goal
        """
        full_joint_low_level_action_plan = []
        hl_plan, cost = self.get_hl_plan(start_state, h_fn)
        curr_state = start_state
        prev_h = h_fn(start_state, debug=False)
        total_cost = 0
        for joint_hl_action, curr_goal_state in hl_plan:
            assert all([type(a) is HighLevelAction for a in joint_hl_action])
            hl_action_plan, curr_state, hl_action_cost = self.perform_hl_action(joint_hl_action, curr_state)
            full_joint_low_level_action_plan.extend(hl_action_plan)
            total_cost += hl_action_cost
            assert curr_state == curr_goal_state

            curr_h = h_fn(curr_state, debug=False)
            self.mlp.check_heuristic_consistency(curr_h, prev_h, total_cost)
            prev_h = curr_h
        assert total_cost == cost == len(full_joint_low_level_action_plan), "{} vs {} vs {}" \
            .format(total_cost, cost, len(full_joint_low_level_action_plan))
        return full_joint_low_level_action_plan, cost

    def get_hl_plan(self, start_state, h_fn, debug=False):
        expand_fn = lambda state: self.get_successor_states(state)
        goal_fn = lambda state: state.num_orders_remaining == 0
        heuristic_fn = lambda state: h_fn(state)

        search_problem = SearchTree(start_state, goal_fn, expand_fn, heuristic_fn, debug=debug)
        hl_plan, cost = search_problem.A_star_graph_search(info=True)
        return hl_plan[1:], cost

    def _advance_motion_goal_indices(self, curr_plan_indices, plan_lengths):
        """Advance indices for agents current motion goals
        based on who finished their motion goal this round"""
        idx0, idx1 = curr_plan_indices
        if plan_lengths[0] == plan_lengths[1]:
            return idx0 + 1, idx1 + 1

        who_finished = np.argmin(plan_lengths)
        if who_finished == 0:
            return idx0 + 1, idx1
        elif who_finished == 1:
            return idx0, idx1 + 1









