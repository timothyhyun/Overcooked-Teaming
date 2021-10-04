import itertools, os
import numpy as np
import pickle, time

import sys, os
import pickle
sys.path.insert(0, "../../")

from overcooked_ai_py.utils import pos_distance, manhattan_distance
from overcooked_ai_py.planning.search import SearchTree, Graph
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, PlayerState, OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.data.planners import load_saved_action_manager, PLANNERS_DIR


class LowLevel_MotionPlanner(object):
    def __init__(self, mdp, player_index, counter_goals=[]):
        self.mdp = mdp
        self.player_index = player_index

        # If positions facing counters should be 
        # allowed as motion goals
        # counter_goals = [(2,1), (2,2), (2,3)]
        self.counter_goals = counter_goals

        # Graph problem that solves shortest path problem
        # between any position & orientation start-goal pair
        self.graph_problem = self._graph_from_grid()
        self.motion_goals_for_pos = self._get_goal_dict()


        self.all_plans = self._populate_all_plans()
        # print("self.all_plans", self.all_plans)
        self.populate_plans_to_onions()

    def populate_plans_to_onions(self):
        self.pot_locations = self.mdp.get_pot_locations()
        self.onion_dispenser_locations = self.mdp.get_onion_dispenser_locations()
        self.counter_locations = self.mdp.get_counter_locations()
        self.dish_dispenser_locations = self.mdp.get_dish_dispenser_locations()
        self.serving_locations = self.mdp.get_serving_locations()


        valid_player_pos_and_ors = self.mdp.get_valid_player_positions_and_orientations()

        # MAKE POT PLANS
        self.plans_to_pots = {}
        pot_goals = [(3,0),(4,1)]
        pot_locations_to_player_goals = {(3,0): ((3,1), (0,-1)),
                                         (4,1): ((3,1), (1, 0))}


        for pot_loc in pot_goals:
            pot_pos_and_or = pot_locations_to_player_goals[pot_loc]
            self.plans_to_pots[pot_loc] = {}
            for start_pos_and_or in valid_player_pos_and_ors:
                plan_key = (start_pos_and_or, pot_pos_and_or)
                if plan_key not in self.all_plans:
                    continue

                action_plan, pos_and_or_path, plan_cost = self.get_plan(start_pos_and_or, pot_pos_and_or)
                self.plans_to_pots[pot_loc][start_pos_and_or] = {}
                self.plans_to_pots[pot_loc][start_pos_and_or]['action_plan'] = action_plan
                self.plans_to_pots[pot_loc][start_pos_and_or]['pos_and_or_path'] = pos_and_or_path
                self.plans_to_pots[pot_loc][start_pos_and_or]['plan_cost'] = plan_cost

        # MAKE ONION DISPENSER PLANS
        self.plans_to_onion_dispensers = {}
        onion_dispenser_goals = [(0,1), (0,2)]
        onion_dispenser_locations_to_player_goals = {(0,1): ((1, 1), (-1,0)),
                                         (0,2): ((1,2), (-1, 0))}

        for od_loc in onion_dispenser_goals:
            od_pos_and_or = onion_dispenser_locations_to_player_goals[od_loc]
            self.plans_to_onion_dispensers[od_loc] = {}
            for start_pos_and_or in valid_player_pos_and_ors:
                plan_key = (start_pos_and_or, od_pos_and_or)
                if plan_key not in self.all_plans:
                    continue

                action_plan, pos_and_or_path, plan_cost = self.get_plan(start_pos_and_or, od_pos_and_or)
                self.plans_to_onion_dispensers[od_loc][start_pos_and_or] = {}
                self.plans_to_onion_dispensers[od_loc][start_pos_and_or]['action_plan'] = action_plan
                self.plans_to_onion_dispensers[od_loc][start_pos_and_or]['pos_and_or_path'] = pos_and_or_path
                self.plans_to_onion_dispensers[od_loc][start_pos_and_or]['plan_cost'] = plan_cost

        # MAKE DISH DISPENSER PLANS
        self.plans_to_dish_dispensers = {}
        dish_dispenser_goals = [(0, 3)]
        dish_dispenser_locations_to_player_goals = {(0, 3): ((1, 3), (-1, 0))}

        for dd_loc in dish_dispenser_goals:
            dd_pos_and_or = dish_dispenser_locations_to_player_goals[dd_loc]
            self.plans_to_dish_dispensers[dd_loc] = {}
            for start_pos_and_or in valid_player_pos_and_ors:
                plan_key = (start_pos_and_or, dd_pos_and_or)
                if plan_key not in self.all_plans:
                    continue

                action_plan, pos_and_or_path, plan_cost = self.get_plan(start_pos_and_or, dd_pos_and_or)
                self.plans_to_dish_dispensers[dd_loc][start_pos_and_or] = {}
                self.plans_to_dish_dispensers[dd_loc][start_pos_and_or]['action_plan'] = action_plan
                self.plans_to_dish_dispensers[dd_loc][start_pos_and_or]['pos_and_or_path'] = pos_and_or_path
                self.plans_to_dish_dispensers[dd_loc][start_pos_and_or]['plan_cost'] = plan_cost

        # MAKE SERVER PLANS
        self.plans_to_server = {}
        server_goals = [(3,4)]
        server_locations_to_player_goals = {(3,4): ((3, 3), (0, 1))}

        for serve_loc in server_goals:
            serve_pos_and_or = server_locations_to_player_goals[serve_loc]
            self.plans_to_server[serve_loc] = {}
            for start_pos_and_or in valid_player_pos_and_ors:
                plan_key = (start_pos_and_or, serve_pos_and_or)
                if plan_key not in self.all_plans:
                    continue

                action_plan, pos_and_or_path, plan_cost = self.get_plan(start_pos_and_or, serve_pos_and_or)
                self.plans_to_server[serve_loc][start_pos_and_or] = {}
                self.plans_to_server[serve_loc][start_pos_and_or]['action_plan'] = action_plan
                self.plans_to_server[serve_loc][start_pos_and_or]['pos_and_or_path'] = pos_and_or_path
                self.plans_to_server[serve_loc][start_pos_and_or]['plan_cost'] = plan_cost



        # MAKE PASSING COUNTER PLANS
        self.plans_to_passing_counters = {}
        passing_counter_goals = [(2, 1), (2,2), (2, 3)]
        if self.player_index == 0:
            passing_counter_locations_to_player_goals = {(2, 1): ((3, 1), (-1, 0)),
                                                        (2, 2): ((3, 2), (-1, 0)),
                                                        (2, 3): ((3, 3), (-1, 0))}
        else:
            passing_counter_locations_to_player_goals = {(2, 1): ((1, 1), (1, 0)),
                                                        (2, 2): ((1, 2), (1, 0)),
                                                        (2, 3): ((1, 3), (1, 0))}


        for pc_loc in passing_counter_goals:
            pc_pos_and_or = passing_counter_locations_to_player_goals[pc_loc]
            self.plans_to_passing_counters[pc_loc] = {}
            for start_pos_and_or in valid_player_pos_and_ors:
                plan_key = (start_pos_and_or, pc_pos_and_or)
                if plan_key not in self.all_plans:
                    continue

                action_plan, pos_and_or_path, plan_cost = self.get_plan(start_pos_and_or, pc_pos_and_or)
                self.plans_to_passing_counters[pc_loc][start_pos_and_or] = {}
                self.plans_to_passing_counters[pc_loc][start_pos_and_or]['action_plan'] = action_plan
                if pc_loc == (2,1) and start_pos_and_or == ((3, 1), (0, -1)):
                    # print('here')
                    action_plan = [(-1,0), 'interact']
                    self.plans_to_passing_counters[pc_loc][start_pos_and_or]['action_plan'] = action_plan

                self.plans_to_passing_counters[pc_loc][start_pos_and_or]['pos_and_or_path'] = pos_and_or_path
                self.plans_to_passing_counters[pc_loc][start_pos_and_or]['plan_cost'] = plan_cost

        # self.plans_to_passing_counters = {(2, 1): {((3, 1), (0, -1)): {'action_plan': [(-1,0), 'interact'], 'pos_and_or_path': [((3, 1), (-1, 0)), ((3, 1), (-1, 0))], 'plan_cost': 2}, ((3, 1), (0, 1)): {'action_plan': [(-1, 0), 'interact'], 'pos_and_or_path': [((3, 1), (-1, 0)), ((3, 1), (-1, 0))], 'plan_cost': 2}, ((3, 1), (1, 0)): {'action_plan': [(-1, 0), 'interact'], 'pos_and_or_path': [((3, 1), (-1, 0)), ((3, 1), (-1, 0))], 'plan_cost': 2}, ((3, 1), (-1, 0)): {'action_plan': ['interact'], 'pos_and_or_path': [((3, 1), (-1, 0))], 'plan_cost': 1}, ((3, 2), (0, -1)): {'action_plan': [(0, -1), (-1, 0), 'interact'], 'pos_and_or_path': [((3, 1), (0, -1)), ((3, 1), (-1, 0)), ((3, 1), (-1, 0))], 'plan_cost': 3}, ((3, 2), (0, 1)): {'action_plan': [(0, -1), (-1, 0), 'interact'], 'pos_and_or_path': [((3, 1), (0, -1)), ((3, 1), (-1, 0)), ((3, 1), (-1, 0))], 'plan_cost': 3}, ((3, 2), (1, 0)): {'action_plan': [(0, -1), (-1, 0), 'interact'], 'pos_and_or_path': [((3, 1), (0, -1)), ((3, 1), (-1, 0)), ((3, 1), (-1, 0))], 'plan_cost': 3}, ((3, 2), (-1, 0)): {'action_plan': [(0, -1), (-1, 0), 'interact'], 'pos_and_or_path': [((3, 1), (0, -1)), ((3, 1), (-1, 0)), ((3, 1), (-1, 0))], 'plan_cost': 3}, ((3, 3), (0, -1)): {'action_plan': [(0, -1), (0, -1), (-1, 0), 'interact'], 'pos_and_or_path': [((3, 2), (0, -1)), ((3, 1), (0, -1)), ((3, 1), (-1, 0)), ((3, 1), (-1, 0))], 'plan_cost': 4}, ((3, 3), (0, 1)): {'action_plan': [(0, -1), (0, -1), (-1, 0), 'interact'], 'pos_and_or_path': [((3, 2), (0, -1)), ((3, 1), (0, -1)), ((3, 1), (-1, 0)), ((3, 1), (-1, 0))], 'plan_cost': 4}, ((3, 3), (1, 0)): {'action_plan': [(0, -1), (0, -1), (-1, 0), 'interact'], 'pos_and_or_path': [((3, 2), (0, -1)), ((3, 1), (0, -1)), ((3, 1), (-1, 0)), ((3, 1), (-1, 0))], 'plan_cost': 4}, ((3, 3), (-1, 0)): {'action_plan': [(0, -1), (0, -1), (-1, 0), 'interact'], 'pos_and_or_path': [((3, 2), (0, -1)), ((3, 1), (0, -1)), ((3, 1), (-1, 0)), ((3, 1), (-1, 0))], 'plan_cost': 4}}, (2, 2): {((3, 1), (0, -1)): {'action_plan': [(0, 1), (-1, 0), 'interact'], 'pos_and_or_path': [((3, 2), (0, 1)), ((3, 2), (-1, 0)), ((3, 2), (-1, 0))], 'plan_cost': 3}, ((3, 1), (0, 1)): {'action_plan': [(0, 1), (-1, 0), 'interact'], 'pos_and_or_path': [((3, 2), (0, 1)), ((3, 2), (-1, 0)), ((3, 2), (-1, 0))], 'plan_cost': 3}, ((3, 1), (1, 0)): {'action_plan': [(0, 1), (-1, 0), 'interact'], 'pos_and_or_path': [((3, 2), (0, 1)), ((3, 2), (-1, 0)), ((3, 2), (-1, 0))], 'plan_cost': 3}, ((3, 1), (-1, 0)): {'action_plan': [(0, 1), (-1, 0), 'interact'], 'pos_and_or_path': [((3, 2), (0, 1)), ((3, 2), (-1, 0)), ((3, 2), (-1, 0))], 'plan_cost': 3}, ((3, 2), (0, -1)): {'action_plan': [(-1, 0), 'interact'], 'pos_and_or_path': [((3, 2), (-1, 0)), ((3, 2), (-1, 0))], 'plan_cost': 2}, ((3, 2), (0, 1)): {'action_plan': [(-1, 0), 'interact'], 'pos_and_or_path': [((3, 2), (-1, 0)), ((3, 2), (-1, 0))], 'plan_cost': 2}, ((3, 2), (1, 0)): {'action_plan': [(-1, 0), 'interact'], 'pos_and_or_path': [((3, 2), (-1, 0)), ((3, 2), (-1, 0))], 'plan_cost': 2}, ((3, 2), (-1, 0)): {'action_plan': ['interact'], 'pos_and_or_path': [((3, 2), (-1, 0))], 'plan_cost': 1}, ((3, 3), (0, -1)): {'action_plan': [(0, -1), (-1, 0), 'interact'], 'pos_and_or_path': [((3, 2), (0, -1)), ((3, 2), (-1, 0)), ((3, 2), (-1, 0))], 'plan_cost': 3}, ((3, 3), (0, 1)): {'action_plan': [(0, -1), (-1, 0), 'interact'], 'pos_and_or_path': [((3, 2), (0, -1)), ((3, 2), (-1, 0)), ((3, 2), (-1, 0))], 'plan_cost': 3}, ((3, 3), (1, 0)): {'action_plan': [(0, -1), (-1, 0), 'interact'], 'pos_and_or_path': [((3, 2), (0, -1)), ((3, 2), (-1, 0)), ((3, 2), (-1, 0))], 'plan_cost': 3}, ((3, 3), (-1, 0)): {'action_plan': [(0, -1), (-1, 0), 'interact'], 'pos_and_or_path': [((3, 2), (0, -1)), ((3, 2), (-1, 0)), ((3, 2), (-1, 0))], 'plan_cost': 3}}, (2, 3): {((3, 1), (0, -1)): {'action_plan': [(0, 1), (0, 1), (-1, 0), 'interact'], 'pos_and_or_path': [((3, 2), (0, 1)), ((3, 3), (0, 1)), ((3, 3), (-1, 0)), ((3, 3), (-1, 0))], 'plan_cost': 4}, ((3, 1), (0, 1)): {'action_plan': [(0, 1), (0, 1), (-1, 0), 'interact'], 'pos_and_or_path': [((3, 2), (0, 1)), ((3, 3), (0, 1)), ((3, 3), (-1, 0)), ((3, 3), (-1, 0))], 'plan_cost': 4}, ((3, 1), (1, 0)): {'action_plan': [(0, 1), (0, 1), (-1, 0), 'interact'], 'pos_and_or_path': [((3, 2), (0, 1)), ((3, 3), (0, 1)), ((3, 3), (-1, 0)), ((3, 3), (-1, 0))], 'plan_cost': 4}, ((3, 1), (-1, 0)): {'action_plan': [(0, 1), (0, 1), (-1, 0), 'interact'], 'pos_and_or_path': [((3, 2), (0, 1)), ((3, 3), (0, 1)), ((3, 3), (-1, 0)), ((3, 3), (-1, 0))], 'plan_cost': 4}, ((3, 2), (0, -1)): {'action_plan': [(0, 1), (-1, 0), 'interact'], 'pos_and_or_path': [((3, 3), (0, 1)), ((3, 3), (-1, 0)), ((3, 3), (-1, 0))], 'plan_cost': 3}, ((3, 2), (0, 1)): {'action_plan': [(0, 1), (-1, 0), 'interact'], 'pos_and_or_path': [((3, 3), (0, 1)), ((3, 3), (-1, 0)), ((3, 3), (-1, 0))], 'plan_cost': 3}, ((3, 2), (1, 0)): {'action_plan': [(0, 1), (-1, 0), 'interact'], 'pos_and_or_path': [((3, 3), (0, 1)), ((3, 3), (-1, 0)), ((3, 3), (-1, 0))], 'plan_cost': 3}, ((3, 2), (-1, 0)): {'action_plan': [(0, 1), (-1, 0), 'interact'], 'pos_and_or_path': [((3, 3), (0, 1)), ((3, 3), (-1, 0)), ((3, 3), (-1, 0))], 'plan_cost': 3}, ((3, 3), (0, -1)): {'action_plan': [(-1, 0), 'interact'], 'pos_and_or_path': [((3, 3), (-1, 0)), ((3, 3), (-1, 0))], 'plan_cost': 2}, ((3, 3), (0, 1)): {'action_plan': [(-1, 0), 'interact'], 'pos_and_or_path': [((3, 3), (-1, 0)), ((3, 3), (-1, 0))], 'plan_cost': 2}, ((3, 3), (1, 0)): {'action_plan': [(-1, 0), 'interact'], 'pos_and_or_path': [((3, 3), (-1, 0)), ((3, 3), (-1, 0))], 'plan_cost': 2}, ((3, 3), (-1, 0)): {'action_plan': ['interact'], 'pos_and_or_path': [((3, 3), (-1, 0))], 'plan_cost': 1}}}


        # print("self.plans_to_passing_counters", self.plans_to_passing_counters)


    def get_plan(self, start_pos_and_or, goal_pos_and_or):
        """
        Returns pre-computed plan from initial agent position
        and orientation to a goal position and orientation.

        Args:
            start_pos_and_or (tuple): starting (pos, or) tuple
            goal_pos_and_or (tuple): goal (pos, or) tuple
        """
        plan_key = (start_pos_and_or, goal_pos_and_or)
        action_plan, pos_and_or_path, plan_cost = self.all_plans[plan_key]
        return action_plan, pos_and_or_path, plan_cost

    def _populate_all_plans(self):
        """Pre-computes all valid plans"""
        all_plans = {}
        valid_pos_and_ors = self.mdp.get_valid_player_positions_and_orientations()
        valid_motion_goals = filter(self.is_valid_motion_goal, valid_pos_and_ors)
        for start_motion_state, goal_motion_state in itertools.product(valid_pos_and_ors, valid_motion_goals):
            if not self.is_valid_motion_start_goal_pair(start_motion_state, goal_motion_state):
                continue
            action_plan, pos_and_or_path, plan_cost = self._compute_plan(start_motion_state, goal_motion_state)
            plan_key = (start_motion_state, goal_motion_state)
            all_plans[plan_key] = (action_plan, pos_and_or_path, plan_cost)
        return all_plans

    def is_valid_motion_goal(self, goal_pos_and_or):
        """Checks that desired single-agent goal state (position and orientation)
        is reachable and is facing a terrain feature"""
        goal_position, goal_orientation = goal_pos_and_or
        if goal_position not in self.mdp.get_valid_player_positions():
            return False

        # Restricting goals to be facing a terrain feature
        pos_of_facing_terrain = Action.move_in_direction(goal_position, goal_orientation)
        facing_terrain_type = self.mdp.get_terrain_type_at_pos(pos_of_facing_terrain)
        if facing_terrain_type == ' ' or (facing_terrain_type == 'X' and pos_of_facing_terrain not in self.counter_goals):
            return False
        return True

    def is_valid_motion_start_goal_pair(self, start_pos_and_or, goal_pos_and_or, debug=False):
        if not self.is_valid_motion_goal(goal_pos_and_or):
            return False
        if not self.positions_are_connected(start_pos_and_or, goal_pos_and_or):
            return False
        return True

    def positions_are_connected(self, start_pos_and_or, goal_pos_and_or):
        return self.graph_problem.are_in_same_cc(start_pos_and_or, goal_pos_and_or)

    def _compute_plan(self, start_motion_state, goal_motion_state):
        """Computes optimal action plan for single agent movement

        Args:
            start_motion_state (tuple): starting positions and orientations
            positions_plan (list): positions path followed by agent
            goal_motion_state (tuple): goal positions and orientations
        """
        assert self.is_valid_motion_start_goal_pair(start_motion_state, goal_motion_state)
        positions_plan = self._get_position_plan_from_graph(start_motion_state, goal_motion_state)
        action_plan, pos_and_or_path, plan_length = self.action_plan_from_positions(positions_plan, start_motion_state,
                                                                                    goal_motion_state)
        return action_plan, pos_and_or_path, plan_length

    def _get_position_plan_from_graph(self, start_node, end_node):
        """Recovers positions to be reached by agent after the start node to reach the end node"""
        node_path = self.graph_problem.get_node_path(start_node, end_node)
        assert node_path[0] == start_node and node_path[-1] == end_node
        positions_plan = [state_node[0] for state_node in node_path[1:]]
        return positions_plan

    def action_plan_from_positions(self, position_list, start_motion_state, goal_motion_state):
        """
        Recovers an action plan reaches the goal motion position and orientation, and executes
        and interact action.

        Args:
            position_list (list): list of positions to be reached after the starting position
                                  (does not include starting position, but includes ending position)
            start_motion_state (tuple): starting position and orientation
            goal_motion_state (tuple): goal position and orientation

        Returns:
            action_plan (list): list of actions to reach goal state
            pos_and_or_path (list): list of (pos, or) pairs visited during plan execution
                                    (not including start, but including goal)
        """
        goal_position, goal_orientation = goal_motion_state
        action_plan, pos_and_or_path = [], []
        position_to_go = list(position_list)
        curr_pos, curr_or = start_motion_state

        # Get agent to goal position
        while position_to_go and curr_pos != goal_position:
            next_pos = position_to_go.pop(0)
            action = Action.determine_action_for_change_in_pos(curr_pos, next_pos)
            action_plan.append(action)
            curr_or = action if action != Action.STAY else curr_or
            pos_and_or_path.append((next_pos, curr_or))
            curr_pos = next_pos

        # Fix agent orientation if necessary
        if curr_or != goal_orientation:
            new_pos, _ = self.mdp._move_if_direction(curr_pos, curr_or, goal_orientation)
            assert new_pos == goal_position
            action_plan.append(goal_orientation)
            pos_and_or_path.append((goal_position, goal_orientation))

        # Add interact action
        action_plan.append(Action.INTERACT)
        pos_and_or_path.append((goal_position, goal_orientation))

        return action_plan, pos_and_or_path, len(action_plan)

    def _graph_from_grid(self):
        """Creates a graph adjacency matrix from an Overcooked MDP class."""
        # State decoder is a dictionary.
        # For all valid player positions and orientations, insert into the state decoder.
        # State decoder takes form -- Counter ID: player ( position, orientation )
        # Position encoder takes form -- player ( position, orientation ): Counter ID
        # Max counter ID = num of graph nodes

        state_decoder = {}
        for state_index, motion_state in enumerate(self.mdp.get_valid_player_positions_and_orientations()):
            state_decoder[state_index] = motion_state

        pos_encoder = {motion_state:state_index for state_index, motion_state in state_decoder.items()}
        num_graph_nodes = len(state_decoder)

        adjacency_matrix = np.zeros((num_graph_nodes, num_graph_nodes))
        for state_index, start_motion_state in state_decoder.items():
            # For each possible next state that the player can be in given an action.
            # action = id, successor_motion_state = (new_pos, new_orientation)
            for action, successor_motion_state in self._get_valid_successor_motion_states(start_motion_state):
                adj_pos_index = pos_encoder[successor_motion_state]
                adjacency_matrix[state_index][adj_pos_index] = self._graph_action_cost(action)
                # An action can take you from one state to the next, given the cost of the action.

        return Graph(adjacency_matrix, pos_encoder, state_decoder)

    def _graph_action_cost(self, action):
        """Returns cost of a single-agent action"""
        assert action in Action.ALL_ACTIONS
        return 1

    def _get_valid_successor_motion_states(self, start_motion_state):
        """Get valid motion states one action away from the starting motion state."""
        start_position, start_orientation = start_motion_state
        # For all actions in the input start state, return list of (new_pos, new_orientation) values in the action was taken.
        return [(action, self.mdp._move_if_direction(start_position, start_orientation, action)) for action in Action.ALL_ACTIONS]


    def _get_goal_dict(self):
        """Creates a dictionary of all possible goal states for all possible
        terrain features that the agent might want to interact with."""
        terrain_feature_locations = []
        for terrain_type, pos_list in self.mdp.terrain_pos_dict.items():
            if terrain_type != ' ':
                terrain_feature_locations += pos_list
        return {feature_pos:self._get_possible_motion_goals_for_feature(feature_pos) for feature_pos in terrain_feature_locations}

    def _get_possible_motion_goals_for_feature(self, goal_pos):
        """Returns a list of possible goal positions (and orientations)
        that could be used for motion planning to get to goal_pos"""
        goals = []
        valid_positions = self.mdp.get_valid_player_positions()
        for d in Direction.ALL_DIRECTIONS:
            adjacent_pos = Action.move_in_direction(goal_pos, d)
            if adjacent_pos in valid_positions:
                goal_orientation = Direction.OPPOSITE_DIRECTIONS[d]
                motion_goal = (adjacent_pos, goal_orientation)
                goals.append(motion_goal)
        return goals



class MediumLevel_MotionPlanner(object):
    def __init__(self, mdp, player_index):
        self.mdp = mdp
        self.player_index = player_index

        # If positions facing counters should be
        # allowed as motion goals
        self.low_level_planner = LowLevel_MotionPlanner(self.mdp, self.player_index, counter_goals=[(2,1), (2,2), (2,3)])
        self.get_info_from_ll_planner()


    def get_info_from_ll_planner(self):
        self.all_plans = self.low_level_planner.all_plans

        self.pot_locations = self.mdp.get_pot_locations()
        self.onion_dispenser_locations = self.mdp.get_onion_dispenser_locations()
        self.counter_locations = self.mdp.get_counter_locations()
        self.dish_dispenser_locations = self.mdp.get_dish_dispenser_locations()
        self.serving_locations = self.mdp.get_serving_locations()

        self.plans_to_pots = self.low_level_planner.plans_to_pots
        self.plans_to_onion_dispensers = self.low_level_planner.plans_to_onion_dispensers
        self.plans_to_dish_dispensers = self.low_level_planner.plans_to_dish_dispensers
        self.plans_to_server = self.low_level_planner.plans_to_server
        self.plans_to_passing_counters = self.low_level_planner.plans_to_passing_counters


    def pickup_onion_from_dispenser(self, player_position, player_orientation):

        if self.player_index == 0:
            action_seq = []
            return action_seq

        action_seq = self.plans_to_onion_dispensers[(0, 1)][(player_position, player_orientation)]['action_plan']
        if self.player_index == 1 and len(action_seq) == 0:
            # if (player_position, player_orientation) == ((1, 1), (1, 0)):
            #     action_seq = [(-1, 0), 'interact']
            #
            # if (player_position, player_orientation) == ((1, 2), (1, 0)):
            #     action_seq = [(0, -1),(-1,0), 'interact']
            if player_position == (1, 1):
                if player_orientation == (0, -1): #North
                    action_seq = [(-1, 0), 'interact']
                if player_orientation == (0, 1): #South
                    action_seq = [(-1, 0), 'interact']
                if player_orientation == (1, 0): # EAST
                    action_seq = [(-1, 0), 'interact']
                if player_orientation == (-1, 0): #WEST
                    action_seq = ['interact']


            if player_position == (1, 2):
                if player_orientation == (0, -1): #North
                    action_seq = [(0, -1), (-1, 0), 'interact']

                if player_orientation == (0, 1): #South
                    action_seq = [(0, -1), (-1, 0), 'interact']

                if player_orientation == (1, 0): # EAST
                    action_seq = [(0, -1),(-1,0), 'interact']

                if player_orientation == (-1, 0): #WEST
                    action_seq = [(0, -1),(-1,0), 'interact']

            if player_position == (1, 3):
                if player_orientation == (0, -1): #North
                    action_seq = [(0, -1),(0, -1), (-1, 0), 'interact']

                if player_orientation == (0, 1): #South
                    action_seq = [(0, -1), (0, -1),(-1, 0), 'interact']

                if player_orientation == (1, 0): # EAST
                    action_seq = [(0, -1),(0, -1),(-1,0), 'interact']

                if player_orientation == (-1, 0): #WEST
                    action_seq = [(0, -1),(0, -1),(-1,0), 'interact']


        # print("pickup onion, Action seq", action_seq)
        # print('plans_to_onion_dispensers', self.plans_to_onion_dispensers)

        return action_seq

    def pickup_dish_from_dispenser(self, player_position, player_orientation):

        if self.player_index == 0:
            action_seq = []
            return action_seq
        # print("PICKUP DISH FROM DISPENSER")
        action_seq = self.plans_to_dish_dispensers[(0, 3)][(player_position, player_orientation)]['action_plan']
        if self.player_index == 1 and len(action_seq) == 0:
            # if (player_position, player_orientation) == ((1, 1), (1, 0)):
            #     action_seq = [(0,1), (0,1), (-1,0), 'interact']
            #
            # if (player_position, player_orientation) == ((1, 2), (1, 0)):
            #     action_seq = [(0,1), (-1,0), 'interact']

            if player_position == (1, 1):
                if player_orientation == (0, -1):  # North
                    action_seq = [(0,1), (0,1), (-1,0), 'interact']
                if player_orientation == (0, 1):  # South
                    action_seq = [(0,1), (0,1), (-1,0), 'interact']
                if player_orientation == (1, 0):  # EAST
                    action_seq = [(0,1), (0,1), (-1,0), 'interact']
                if player_orientation == (-1, 0):  # WEST
                    action_seq = [(0,1), (0,1), (-1,0), 'interact']

            if player_position == (1, 2):
                if player_orientation == (0, -1):  # North
                    action_seq = [(0,1), (-1,0), 'interact']

                if player_orientation == (0, 1):  # South
                    action_seq = [(0,1), (-1,0), 'interact']

                if player_orientation == (1, 0):  # EAST
                    action_seq = [(0,1), (-1,0), 'interact']

                if player_orientation == (-1, 0):  # WEST
                    action_seq = [(0,1), (-1,0), 'interact']

            if player_position == (1, 3):
                if player_orientation == (0, -1):  # North
                    action_seq = [(-1, 0), 'interact']

                if player_orientation == (0, 1):  # South
                    action_seq = [(-1, 0), 'interact']

                if player_orientation == (1, 0):  # EAST
                    action_seq = [(-1, 0), 'interact']

                if player_orientation == (-1, 0):  # WEST
                    action_seq = ['interact']

        return action_seq

    def put_onion_on_passing_counter(self, player_position, player_orientation):
        action_seq = self.plans_to_passing_counters[(2, 1)][(player_position, player_orientation)]['action_plan']
        if self.player_index == 1 and len(action_seq) == 0:
            # if (player_position, player_orientation) == ((1, 1), (-1, 0)):
            #     # print('here')
            #     action_seq = [(1, 0), 'interact']
            # if (player_position, player_orientation) == ((1, 2), (-1, 0)):
            #     action_seq = [(0, -1), (1,0), 'interact']
            # if (player_position, player_orientation) == ((1, 2), (1, 0)):
            #     action_seq = [(0, -1), (1,0), 'interact']

            if player_position == (1, 1):
                if player_orientation == (0, -1):  # North
                    action_seq = [(1, 0), 'interact']
                if player_orientation == (0, 1):  # South
                    action_seq = [(1, 0), 'interact']
                if player_orientation == (1, 0):  # EAST
                    action_seq = ['interact']
                if player_orientation == (-1, 0):  # WEST
                    action_seq = [(1, 0), 'interact']

            if player_position == (1, 2):
                if player_orientation == (0, -1):  # North
                    action_seq = [(0, -1), (1,0), 'interact']

                if player_orientation == (0, 1):  # South
                    action_seq = [(0, -1), (1,0), 'interact']

                if player_orientation == (1, 0):  # EAST
                    action_seq = [(0, -1), (1,0), 'interact']

                if player_orientation == (-1, 0):  # WEST
                    action_seq = [(0, -1), (1,0), 'interact']

            if player_position == (1, 3):
                if player_orientation == (0, -1):  # North
                    action_seq = [(0, -1),(0, -1), (1,0), 'interact']

                if player_orientation == (0, 1):  # South
                    action_seq = [(0, -1),(0, -1), (1,0), 'interact']

                if player_orientation == (1, 0):  # EAST
                    action_seq = [(0, -1),(0, -1), (1,0), 'interact']

                if player_orientation == (-1, 0):  # WEST
                    action_seq = [(0, -1),(0, -1), (1,0), 'interact']

        # print('plans_to_passing_counters', self.plans_to_passing_counters)
        # print("put onion on passing counter: Action seq", action_seq)
        return action_seq

    def put_dish_on_passing_counter(self, player_position, player_orientation):
        # print("PUT DISH ON PASSING COUNTER")
        # print('plans_to_passing_counters', self.plans_to_passing_counters)
        # print('player_position, player_orientation', (player_position, player_orientation))
        action_seq = self.plans_to_passing_counters[(2, 2)][(player_position, player_orientation)]['action_plan']
        # print('action seq', action_seq)
        if self.player_index == 1 and len(action_seq) == 0:
            # if (player_position, player_orientation) == ((1, 3), (-1, 0)):
            #     # print('here')
            #     action_seq = [(0,-1),(1,0), 'interact']
            if player_position == (1, 1):
                if player_orientation == (0, -1):  # North
                    action_seq = [(0,1),(1,0), 'interact']
                if player_orientation == (0, 1):  # South
                    action_seq = [(0,1),(1,0), 'interact']
                if player_orientation == (1, 0):  # EAST
                    action_seq = [(0,1),(1,0), 'interact']
                if player_orientation == (-1, 0):  # WEST
                    action_seq = [(0,1),(1,0), 'interact']

            if player_position == (1, 2):
                if player_orientation == (0, -1):  # North
                    action_seq = [(1,0), 'interact']

                if player_orientation == (0, 1):  # South
                    action_seq = [(1,0), 'interact']

                if player_orientation == (1, 0):  # EAST
                    action_seq = ['interact']

                if player_orientation == (-1, 0):  # WEST
                    action_seq = [(1,0), 'interact']

            if player_position == (1, 3):
                if player_orientation == (0, -1):  # North
                    action_seq = [(0,-1),(1,0), 'interact']

                if player_orientation == (0, 1):  # South
                    action_seq = [(0,-1),(1,0), 'interact']

                if player_orientation == (1, 0):  # EAST
                    action_seq = [(0,-1),(1,0), 'interact']

                if player_orientation == (-1, 0):  # WEST
                    action_seq = [(0,-1),(1,0), 'interact']

        return action_seq

    def pickup_object_from_passing_counter(self, player_position, player_orientation, counter_location):
        action_seq = self.plans_to_passing_counters[counter_location][(player_position, player_orientation)]['action_plan']

        # print('player_position, player_orientation, ', (player_position, player_orientation))
        # print('pickup from counter', action_seq)

        if self.player_index == 0 and len(action_seq) == 0:
            # if counter_location == (2, 1) and (player_position, player_orientation) == ((3, 1), (0, -1)):
            #     # print('here')
            #     action_seq = [(-1, 0), 'interact']
            # if counter_location == (2, 1) and (player_position, player_orientation) == ((3, 1), (1,0)):
            #     # print('here')
            #     action_seq = [(-1, 0), 'interact']
            #
            # if counter_location == (2, 2) and (player_position, player_orientation) == ((3, 1), (1,0)):
            #     action_seq = [(0, 1), (-1,0), 'interact']
            #
            # if counter_location == (2, 2) and (player_position, player_orientation) == ((3, 3), (0,1)):
            #     action_seq = [(0, -1), (-1,0), 'interact']
            #
            # if counter_location == (2,1) and (player_position, player_orientation) == ((3, 3), (0,1)):
            #     action_seq = [(0, -1), (0,-1), (-1,0), 'interact']
            if counter_location == (2, 1):
                if player_position == (3, 1):
                    if player_orientation == (0, -1):  # North
                        action_seq = [(-1, 0), 'interact']
                    if player_orientation == (0, 1):  # South
                        action_seq = [(-1, 0), 'interact']
                    if player_orientation == (1, 0):  # EAST
                        action_seq = [(-1, 0), 'interact']
                    if player_orientation == (-1, 0):  # WEST
                        action_seq = ['interact']

                if player_position == (3, 2):
                    if player_orientation == (0, -1):  # North
                        action_seq = [(0, -1), (-1, 0), 'interact']

                    if player_orientation == (0, 1):  # South
                        action_seq = [(0, -1), (-1, 0), 'interact']

                    if player_orientation == (1, 0):  # EAST
                        action_seq = [(0, -1), (-1, 0), 'interact']

                    if player_orientation == (-1, 0):  # WEST
                        action_seq = [(0, -1), (-1, 0), 'interact']

                if player_position == (3, 3):
                    if player_orientation == (0, -1):  # North
                        action_seq = [(0, -1), (0, -1), (-1, 0), 'interact']

                    if player_orientation == (0, 1):  # South
                        action_seq = [(0, -1), (0, -1), (-1, 0), 'interact']

                    if player_orientation == (1, 0):  # EAST
                        action_seq = [(0, -1), (0, -1), (-1, 0), 'interact']

                    if player_orientation == (-1, 0):  # WEST
                        action_seq = [(0, -1), (0, -1), (-1, 0), 'interact']

            if counter_location == (2, 2):
                if player_position == (3, 1):
                    if player_orientation == (0, -1):  # North
                        action_seq = [(0, 1), (-1, 0), 'interact']
                    if player_orientation == (0, 1):  # South
                        action_seq = [(0, 1), (-1, 0), 'interact']
                    if player_orientation == (1, 0):  # EAST
                        action_seq = [(0, 1), (-1, 0), 'interact']
                    if player_orientation == (-1, 0):  # WEST
                        action_seq = [(0, 1), (-1, 0), 'interact']

                if player_position == (3, 2):
                    if player_orientation == (0, -1):  # North
                        action_seq = [(-1, 0), 'interact']
                    if player_orientation == (0, 1):  # South
                        action_seq = [(-1, 0), 'interact']
                    if player_orientation == (1, 0):  # EAST
                        action_seq = [(-1, 0), 'interact']
                    if player_orientation == (-1, 0):  # WEST
                        action_seq = ['interact']

                if player_position == (3, 3):
                    if player_orientation == (0, -1):  # North
                        action_seq = [(0, -1), (-1,0), 'interact']

                    if player_orientation == (0, 1):  # South
                        action_seq = [(0, -1), (-1, 0), 'interact']

                    if player_orientation == (1, 0):  # EAST
                        action_seq = [(0, -1), (-1, 0), 'interact']

                    if player_orientation == (-1, 0):  # WEST
                        action_seq = [(0, -1), (-1, 0), 'interact']

        # if len(action_seq) == 0:
        #     action_seq.append(Action.INTERACT)
        return action_seq

    def put_onion_in_pot(self, player_position, player_orientation, pot_location):
        if self.player_index == 1:
            return []
        # print("PUT ONION IN POT")

        # print('player_position, player_orientation, pot_locatio', (player_position, player_orientation, pot_location))
        action_seq = self.plans_to_pots[pot_location][(player_position, player_orientation)]['action_plan']
        # print('initial action seq', action_seq)
        if self.player_index == 0 and len(action_seq) == 0:
            # if pot_location == (3, 0) and (player_position, player_orientation) == ((3, 1), (-1, 0)):
            #     action_seq = [(0, -1), 'interact']
            # if pot_location == (4,1) and (player_position, player_orientation) == ((3, 1), (-1, 0)):
            #     action_seq = [(1,0), 'interact']
            if pot_location == (3, 0):
                if player_position == (3, 1):
                    if player_orientation == (0, -1):  # North
                        action_seq = ['interact']
                    if player_orientation == (0, 1):  # South
                        action_seq = [(0, -1), 'interact']
                    if player_orientation == (1, 0):  # EAST
                        action_seq = [(0, -1), 'interact']
                    if player_orientation == (-1, 0):  # WEST
                        action_seq = [(0, -1), 'interact']

                if player_position == (3, 2):
                    if player_orientation == (0, -1):  # North
                        action_seq = [(0, -1), 'interact']

                    if player_orientation == (0, 1):  # South
                        action_seq = [(0, -1), 'interact']

                    if player_orientation == (1, 0):  # EAST
                        action_seq = [(0, -1),  'interact']

                    if player_orientation == (-1, 0):  # WEST
                        action_seq = [(0, -1), 'interact']

                if player_position == (3, 3):
                    if player_orientation == (0, -1):  # North
                        action_seq = [(0, -1), (0, -1), 'interact']

                    if player_orientation == (0, 1):  # South
                        action_seq = [(0, -1), (0, -1), 'interact']

                    if player_orientation == (1, 0):  # EAST
                        action_seq = [(0, -1), (0, -1), 'interact']

                    if player_orientation == (-1, 0):  # WEST
                        action_seq = [(0, -1), (0, -1), 'interact']

            if pot_location == (4,1):
                if player_position == (3, 1):
                    if player_orientation == (0, -1):  # North
                        action_seq = [(1,0), 'interact']
                    if player_orientation == (0, 1):  # South
                        action_seq = [(1,0), 'interact']
                    if player_orientation == (1, 0):  # EAST
                        action_seq = ['interact']
                    if player_orientation == (-1, 0):  # WEST
                        action_seq = [(1,0), 'interact']

                if player_position == (3, 2):
                    if player_orientation == (0, -1):  # North
                        action_seq = [(0, -1), (1,0), 'interact']
                    if player_orientation == (0, 1):  # South
                        action_seq = [(0, -1), (1,0), 'interact']
                    if player_orientation == (1, 0):  # EAST
                        action_seq = [(0, -1), (1,0), 'interact']
                    if player_orientation == (-1, 0):  # WEST
                        action_seq = [(0, -1), (1,0), 'interact']

                if player_position == (3, 3):
                    if player_orientation == (0, -1):  # North
                        action_seq = [(0, -1), (0, -1), (1,0), 'interact']

                    if player_orientation == (0, 1):  # South
                        action_seq = [(0, -1), (0, -1), (1,0), 'interact']

                    if player_orientation == (1, 0):  # EAST
                        action_seq = [(0, -1), (0, -1), (1,0), 'interact']

                    if player_orientation == (-1, 0):  # WEST
                        action_seq = [(0, -1), (0, -1), (1,0), 'interact']

        return action_seq

    def pickup_soup_from_pot(self, player_position, player_orientation, pot_location):

        if self.player_index == 1:
            return []
        # print("PICKUP SOUP FROM POT")
        # print('plans_to_pots', self.plans_to_pots)
        action_seq = self.plans_to_pots[pot_location][(player_position, player_orientation)]['action_plan']
        if self.player_index == 0 and len(action_seq) == 0:
            # if pot_location == (3, 0) and (player_position, player_orientation) == ((3, 1), (-1, 0)):
            #     # print('here')
            #     action_seq = [(0, -1), 'interact']
            # if pot_location == (4,1) and (player_position, player_orientation) == ((3, 1), (-1, 0)):
            #     action_seq = [(1,0), 'interact']
            #
            # if pot_location == (3,0) and (player_position, player_orientation) == ((3, 2), (-1, 0)):
            #     action_seq = [(0, -1), 'interact']
            #
            # if pot_location == (4,1) and (player_position, player_orientation) == ((3, 2), (-1, 0)):
            #     action_seq = [(0, -1), (1,0), 'interact']
            if pot_location == (3, 0):
                if player_position == (3, 1):
                    if player_orientation == (0, -1):  # North
                        action_seq = ['interact']
                    if player_orientation == (0, 1):  # South
                        action_seq = [(0, -1), 'interact']
                    if player_orientation == (1, 0):  # EAST
                        action_seq = [(0, -1), 'interact']
                    if player_orientation == (-1, 0):  # WEST
                        action_seq = [(0, -1), 'interact']

                if player_position == (3, 2):
                    if player_orientation == (0, -1):  # North
                        action_seq = [(0, -1), 'interact']

                    if player_orientation == (0, 1):  # South
                        action_seq = [(0, -1), 'interact']

                    if player_orientation == (1, 0):  # EAST
                        action_seq = [(0, -1),  'interact']

                    if player_orientation == (-1, 0):  # WEST
                        action_seq = [(0, -1), 'interact']

                if player_position == (3, 3):
                    if player_orientation == (0, -1):  # North
                        action_seq = [(0, -1), (0, -1), 'interact']

                    if player_orientation == (0, 1):  # South
                        action_seq = [(0, -1), (0, -1), 'interact']

                    if player_orientation == (1, 0):  # EAST
                        action_seq = [(0, -1), (0, -1), 'interact']

                    if player_orientation == (-1, 0):  # WEST
                        action_seq = [(0, -1), (0, -1), 'interact']

            if pot_location == (4,1):
                if player_position == (3, 1):
                    if player_orientation == (0, -1):  # North
                        action_seq = [(1,0), 'interact']
                    if player_orientation == (0, 1):  # South
                        action_seq = [(1,0), 'interact']
                    if player_orientation == (1, 0):  # EAST
                        action_seq = ['interact']
                    if player_orientation == (-1, 0):  # WEST
                        action_seq = [(1,0), 'interact']

                if player_position == (3, 2):
                    if player_orientation == (0, -1):  # North
                        action_seq = [(0, -1), (1,0), 'interact']
                    if player_orientation == (0, 1):  # South
                        action_seq = [(0, -1), (1,0), 'interact']
                    if player_orientation == (1, 0):  # EAST
                        action_seq = [(0, -1), (1,0), 'interact']
                    if player_orientation == (-1, 0):  # WEST
                        action_seq = [(0, -1), (1,0), 'interact']

                if player_position == (3, 3):
                    if player_orientation == (0, -1):  # North
                        action_seq = [(0, -1), (0, -1), (1,0), 'interact']

                    if player_orientation == (0, 1):  # South
                        action_seq = [(0, -1), (0, -1), (1,0), 'interact']

                    if player_orientation == (1, 0):  # EAST
                        action_seq = [(0, -1), (0, -1), (1,0), 'interact']

                    if player_orientation == (-1, 0):  # WEST
                        action_seq = [(0, -1), (0, -1), (1,0), 'interact']

        return action_seq

    def serve_soup(self, player_position, player_orientation):

        if self.player_index == 1:
            return []
        # print('self.plans_to_server', self.plans_to_server)
        action_seq = self.plans_to_server[(3,4)][(player_position, player_orientation)]['action_plan']

        # if (player_position, player_orientation) == ((3, 1), (0,-1)):
        #     action_seq = [(0, 1), (0,1), 'interact']
        #
        # if (player_position, player_orientation) == ((3, 1), (1,0)):
        #     action_seq = [(0, 1), (0,1), 'interact']
        if player_position == (3, 1):
            if player_orientation == (0, -1):  # North
                action_seq = [(0, 1), (0,1), 'interact']
            if player_orientation == (0, 1):  # South
                action_seq = [(0, 1), (0,1), 'interact']
            if player_orientation == (1, 0):  # EAST
                action_seq = [(0, 1), (0,1), 'interact']
            if player_orientation == (-1, 0):  # WEST
                action_seq = [(0, 1), (0,1), 'interact']

        if player_position == (3, 2):
            if player_orientation == (0, -1):  # North
                action_seq = [(0,1), 'interact']
            if player_orientation == (0, 1):  # South
                action_seq = [(0,1), 'interact']
            if player_orientation == (1, 0):  # EAST
                action_seq = [(0,1), 'interact']
            if player_orientation == (-1, 0):  # WEST
                action_seq = [(0,1), 'interact']

        if player_position == (3, 3):
            if player_orientation == (0, -1):  # North
                action_seq = [(0,1), 'interact']

            if player_orientation == (0, 1):  # South
                action_seq = ['interact']

            if player_orientation == (1, 0):  # EAST
                action_seq = [(0,1), 'interact']

            if player_orientation == (-1, 0):  # WEST
                action_seq = [(0,1), 'interact']

        return action_seq









