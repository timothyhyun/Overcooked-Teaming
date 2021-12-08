import itertools, copy
import numpy as np
from functools import reduce
from collections import defaultdict
from overcooked_ai_py.utils import pos_distance, load_from_json
from overcooked_ai_py.data.layouts import read_layout_dict
from overcooked_ai_py.mdp.actions import Action, Direction


class ObjectState(object):
    """
    State of an object in OvercookedGridworld.
    """

    SOUP_TYPES = ['onion', 'tomato']

    def __init__(self, name, position, state=None):
        """
        name (str): The name of the object
        position (int, int): Tuple for the current location of the object.
        state (tuple or None):  
            Extra information about the object. Is None for all objects 
            except soups, for which `state` is a tuple:
            (soup_type, num_items, cook_time)
            where cook_time is how long the soup has been cooking for.
        """
        self.name = name
        self.position = tuple(position)
        if name == 'soup':
            assert len(state) == 3
        self.state = None if state is None else tuple(state)

    def is_valid(self):
        if self.name in ['onion', 'tomato', 'dish']:
            return self.state is None
        elif self.name == 'soup':
            soup_type, num_items, cook_time = self.state
            valid_soup_type = soup_type in self.SOUP_TYPES
            valid_item_num = (1 <= num_items <= 3)
            valid_cook_time = (0 <= cook_time)
            return valid_soup_type and valid_item_num and valid_cook_time
        # Unrecognized object
        return False

    def deepcopy(self):
        return ObjectState(self.name, self.position, self.state)

    def __eq__(self, other):
        return isinstance(other, ObjectState) and \
            self.name == other.name and \
            self.position == other.position and \
            self.state == other.state

    def __hash__(self):
        return hash((self.name, self.position, self.state))

    def __repr__(self):
        if self.state is None:
            return '{}@{}'.format(self.name, self.position)
        return '{}@{} with state {}'.format(
            self.name, self.position, str(self.state))

    def to_dict(self):
        return {
            "name": self.name,
            "position": self.position,
            "state": self.state
        }

    @staticmethod
    def from_dict(obj_dict):
        obj_dict = copy.deepcopy(obj_dict)

        new_obj_dict = {}
        new_obj_dict['name'] = obj_dict['name']
        new_obj_dict['position'] = obj_dict['position']
        new_obj_dict['state'] = None
        if obj_dict['name'] == 'soup':
            soup_state = ('onion', len(obj_dict['_ingredients']), obj_dict['cook_time'])
            new_obj_dict['state'] = soup_state


        return ObjectState(**new_obj_dict)


class PlayerState(object):
    """
    State of a player in OvercookedGridworld.

    position: (x, y) tuple representing the player's location.
    orientation: Direction.NORTH/SOUTH/EAST/WEST representing orientation.
    held_object: ObjectState representing the object held by the player, or
                 None if there is no such object.
    """
    def __init__(self, position, orientation, held_object=None):
        self.position = tuple(position)
        self.orientation = tuple(orientation)
        self.held_object = held_object

        assert self.orientation in Direction.ALL_DIRECTIONS
        if self.held_object is not None:
            assert isinstance(self.held_object, ObjectState)
            assert self.held_object.position == self.position

    @property
    def pos_and_or(self):
        return (self.position, self.orientation)

    def has_object(self):
        return self.held_object is not None

    def get_object(self):
        assert self.has_object()
        return self.held_object

    def set_object(self, obj):
        assert not self.has_object()
        obj.position = self.position
        self.held_object = obj
 
    def remove_object(self):
        assert self.has_object()
        obj = self.held_object
        self.held_object = None
        return obj
    
    def update_pos_and_or(self, new_position, new_orientation):
        self.position = new_position
        self.orientation = new_orientation
        if self.has_object():
            self.get_object().position = new_position

    def deepcopy(self):
        new_obj = None if self.held_object is None else self.held_object.deepcopy()
        return PlayerState(self.position, self.orientation, new_obj)

    def __eq__(self, other):
        return isinstance(other, PlayerState) and \
            self.position == other.position and \
            self.orientation == other.orientation and \
            self.held_object == other.held_object

    def __hash__(self):
        return hash((self.position, self.orientation, self.held_object))

    def __repr__(self):
        return '{} facing {} holding {}'.format(
            self.position, self.orientation, str(self.held_object))
    
    def to_dict(self):
        return {
            "position": self.position,
            "orientation": self.orientation,
            "held_object": self.held_object.to_dict() if self.held_object is not None else None
        }

    @staticmethod
    def from_dict(player_dict):
        player_dict = copy.deepcopy(player_dict)
        if "held_object" not in player_dict:
            held_obj = None
        else:
            held_obj = player_dict["held_object"]
            if held_obj is not None:
                player_dict["held_object"] = ObjectState.from_dict(held_obj)
        return PlayerState(**player_dict)


class OvercookedState(object):
    """A state in OvercookedGridworld."""
    def __init__(self, players, objects, order_list):
        """
        players: List of PlayerStates (order corresponds to player indices).
        objects: Dictionary mapping positions (x, y) to ObjectStates. 
                 NOTE: Does NOT include objects held by players (they are in 
                 the PlayerState objects).
        order_list: Current orders to be delivered

        NOTE: Does not contain time left, which is handled from the environment side.
        """
        for pos, obj in objects.items():
            assert obj.position == pos
        self.players = tuple(players)
        self.objects = objects
        if order_list is not None:
            assert all([o in OvercookedGridworld.ORDER_TYPES for o in order_list])
        self.order_list = order_list

    @property
    def player_positions(self):
        return tuple([player.position for player in self.players])

    @property
    def player_orientations(self):
        return tuple([player.orientation for player in self.players])

    @property
    def players_pos_and_or(self):
        """Returns a ((pos1, or1), (pos2, or2)) tuple"""
        return tuple(zip(*[self.player_positions, self.player_orientations]))

    @property
    def unowned_objects_by_type(self):
        """
        Returns dictionary of (obj_name: ObjState)
        for all objects in the environment, NOT including
        ones held by players.
        """
        objects_by_type = defaultdict(list)
        for pos, obj in self.objects.items():
            objects_by_type[obj.name].append(obj)
        return objects_by_type

    @property
    def player_objects_by_type(self):
        """
        Returns dictionary of (obj_name: ObjState)
        for all objects held by players.
        """
        player_objects = defaultdict(list)
        for player in self.players:
            if player.has_object():
                player_obj = player.get_object()
                player_objects[player_obj.name].append(player_obj)
        return player_objects

    @property
    def all_objects_by_type(self):
        """
        Returns dictionary of (obj_name: ObjState)
        for all objects in the environment, including
        ones held by players.
        """
        all_objs_by_type = self.unowned_objects_by_type.copy()
        all_objs_by_type.update(self.player_objects_by_type)
        return all_objs_by_type

    @property
    def all_objects_list(self):
        all_objects_lists = list(self.all_objects_by_type.values()) + [[], []]
        return reduce(lambda x, y: x + y, all_objects_lists)

    @property
    def curr_order(self):
        return "any" if self.order_list is None else self.order_list[0]

    @property
    def next_order(self):
        return "any" if self.order_list is None else self.order_list[1]

    @property
    def num_orders_remaining(self):
        return np.Inf if self.order_list is None else len(self.order_list)

    def has_object(self, pos):
        return pos in self.objects

    def get_object(self, pos):
        assert self.has_object(pos)
        return self.objects[pos]

    def add_object(self, obj, pos=None):
        if pos is None:
            pos = obj.position

        assert not self.has_object(pos)
        obj.position = pos
        self.objects[pos] = obj

    def remove_object(self, pos):
        assert self.has_object(pos)
        obj = self.objects[pos]
        del self.objects[pos]
        return obj

    @staticmethod
    def from_players_pos_and_or(players_pos_and_or, order_list):
        """
        Make a dummy OvercookedState with no objects based on the passed in player
        positions and orientations and order list
        """
        return OvercookedState(
            [PlayerState(*player_pos_and_or) for player_pos_and_or in players_pos_and_or], 
            objects={}, order_list=order_list)

    @staticmethod
    def from_player_positions(player_positions, order_list):
        """
        Make a dummy OvercookedState with no objects and with players facing
        North based on the passed in player positions and order list
        """
        dummy_pos_and_or = [(pos, Direction.NORTH) for pos in player_positions]
        return OvercookedState.from_players_pos_and_or(dummy_pos_and_or, order_list)

    def deepcopy(self):
        return OvercookedState(
            [player.deepcopy() for player in self.players],
            {pos:obj.deepcopy() for pos, obj in self.objects.items()}, 
            None if self.order_list is None else list(self.order_list))

    def __eq__(self, other):
        order_list_equal = type(self.order_list) == type(other.order_list) and \
            ((self.order_list is None and other.order_list is None) or \
            (type(self.order_list) is list and np.array_equal(self.order_list, other.order_list)))

        return isinstance(other, OvercookedState) and \
            self.players == other.players and \
            set(self.objects.items()) == set(other.objects.items()) and \
            order_list_equal

    def __hash__(self):
        return hash(
            (self.players, tuple(self.objects.values()), tuple(self.order_list))
        )

    def __str__(self):
        return 'Players: {}, Objects: {}, Order list: {}'.format( 
            str(self.players), str(list(self.objects.values())), str(self.order_list))

    def to_dict(self):
        return {
            "players": [p.to_dict() for p in self.players],
            "objects": [obj.to_dict() for obj in self.objects.values()],
            "order_list": self.order_list
        }

    @staticmethod
    def from_dict(state_dict):
        state_dict = copy.deepcopy(state_dict)

        new_state_dict = {}

        new_state_dict["players"] = [PlayerState.from_dict(p) for p in state_dict["players"]]

        # state_dict_objs = list(state_dict["objects"].values())
        object_list = [ObjectState.from_dict(o) for o in state_dict["objects"]]
        new_state_dict["objects"] = { ob.position : ob for ob in object_list }
        new_state_dict["order_list"] = None

        return OvercookedState(**new_state_dict)
    
    @staticmethod
    def from_json(filename):
        return load_from_json(filename)


NO_REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 0,
    "DISH_PICKUP_REWARD": 0,
    "SOUP_PICKUP_REWARD": 0,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
}

BASE_REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0
}


# State, Action Featurization
# 1. Onion placed in empty pot
# 2. Onion placed in partially full pot
# 3. Dish picked up from dispenser if no dishes on counters, and # nearly ready pots > dishes out already
# 4. Soup picked up from ready pot
# 5. Both pots cooking simultaneously
# 6. Serve soup

scale = 2
DP_REW_SHAPING_PARAMS_7 = {
    "ONION_IN_EMPTY_POT_REWARD": 0.78 * scale,
    "ONION_IN_PARTIAL_POT_REWARD": 1.46 * scale,
    "DISH_PICKUP_REWARD": 0.63 * scale,
    "SOUP_PICKUP_FROM_READY_POT_REWARD": 0.76 * scale,
    "BOTH_POTS_FULL_REWARD": 4.93 * scale,
    "SERVE_SOUP_REWARD": 0 * scale,
    "SHARED_COUNTER_REWARD": 0.60 * scale,
}
#
# DP_REW_SHAPING_PARAMS_6 = {
#     "ONION_IN_EMPTY_POT_REWARD": 0.84 * scale,
#     "ONION_IN_PARTIAL_POT_REWARD": 1.56 * scale,
#     "DISH_PICKUP_REWARD": 0.67,
#     "SOUP_PICKUP_FROM_READY_POT_REWARD": 0.80 * scale,
#     "BOTH_POTS_FULL_REWARD": 5.24 * scale,
#     "SERVE_SOUP_REWARD": 0.89 * scale,
# }

# [0.15588226 0.27       0.11882367 0.14941186 0.1323528  0.17352941]
DP_REW_SHAPING_PARAMS_6 = {
    "ONION_IN_EMPTY_POT_REWARD": 1.56 * scale,
    "ONION_IN_PARTIAL_POT_REWARD": 2.70 * scale,
    "DISH_PICKUP_REWARD": 1.19 * scale,
    "SOUP_PICKUP_FROM_READY_POT_REWARD": 1.49 * scale,
    "BOTH_POTS_FULL_REWARD": 1.32 * scale,
    "SERVE_SOUP_REWARD": 1.74 * scale,
}

SP_REW_SHAPING_PARAMS_7 = {
    "ONION_IN_EMPTY_POT_REWARD": 1.48 * scale,
    "ONION_IN_PARTIAL_POT_REWARD": 2.93 * scale,
    "DISH_PICKUP_REWARD": 1.03 * scale,
    "SOUP_PICKUP_FROM_READY_POT_REWARD": 1.83 * scale,
    "BOTH_POTS_FULL_REWARD": 0 * scale,
    "SERVE_SOUP_REWARD": 1.56 * scale,
    "SHARED_COUNTER_REWARD": 1.15 * scale,
}

SP_REW_SHAPING_PARAMS_6 = {
    "ONION_IN_EMPTY_POT_REWARD": 1.69 * scale,
    "ONION_IN_PARTIAL_POT_REWARD": 3.31 * scale,
    "DISH_PICKUP_REWARD": 1.17 * scale,
    "SOUP_PICKUP_FROM_READY_POT_REWARD": 2.07 * scale,
    "BOTH_POTS_FULL_REWARD": 0 * scale,
    "SERVE_SOUP_REWARD": 1.77 * scale,
}


DP_REW_SHAPING_PARAMS_HAND = {
    "ONION_IN_EMPTY_POT_REWARD": 4,
    "ONION_IN_PARTIAL_POT_REWARD": 4,
    "DISH_PICKUP_REWARD": 1,
    "SOUP_PICKUP_FROM_READY_POT_REWARD": 1,
    "BOTH_POTS_FULL_REWARD": 6,
    "SERVE_SOUP_REWARD": 1,
}

BASE_REW_SHAPING_PARAMS = {
    "ONION_IN_EMPTY_POT_REWARD": 3,
    "ONION_IN_PARTIAL_POT_REWARD": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_FROM_READY_POT_REWARD": 4,
    "BOTH_POTS_FULL_REWARD": 6,
    "SERVE_SOUP_REWARD": 1,
    "SHARED_COUNTER_REWARD": 1,
}


class OvercookedGridworld(object):
    """An MDP grid world based off of the Overcooked game."""
    ORDER_TYPES = ObjectState.SOUP_TYPES + ['any']

    def __init__(self, terrain, start_player_positions, start_order_list=None, cook_time=20, num_items_for_soup=3,
                 delivery_reward=20, rew_shaping_params=None, layout_name="unnamed_layout"):
        """
        terrain: a matrix of strings that encode the MDP layout
        layout_name: string identifier of the layout
        start_player_positions: tuple of positions for both players' starting positions
        start_order_list: either a tuple of orders or None if there is not specific list
        cook_time: amount of timesteps required for a soup to cook
        delivery_reward: amount of reward given per delivery
        rew_shaping_params: reward given for completion of specific subgoals
        """
        self.height = len(terrain)
        self.width = len(terrain[0])
        self.shape = (self.width, self.height)
        self.terrain_mtx = terrain
        self.terrain_pos_dict = self._get_terrain_type_pos_dict()
        self.start_player_positions = start_player_positions
        self.num_players = len(start_player_positions)
        self.start_order_list = start_order_list
        self.soup_cooking_time = cook_time
        self.num_items_for_soup = num_items_for_soup
        self.delivery_reward = delivery_reward
        # self.reward_shaping_params = NO_REW_SHAPING_PARAMS if rew_shaping_params is None else rew_shaping_params
        # self.reward_shaping_params = DP_REW_SHAPING_PARAMS_HAND
        self.reward_shaping_params = BASE_REW_SHAPING_PARAMS if rew_shaping_params is None else rew_shaping_params

        # rescale reward_shaping params
        scale = 3/self.reward_shaping_params['ONION_IN_EMPTY_POT_REWARD']
        for keyname in self.reward_shaping_params:
            self.reward_shaping_params[keyname] = self.reward_shaping_params[keyname]*scale

        # print("\n\n\nreward_shaping_params: ", self.reward_shaping_params)
        self.layout_name = layout_name

        self.item_tracking_dict = {}
        self.mean_limbo_time = 0
        self.mean_handoff_time = 0
        self.item_uid_to_location = {}
        self.location_to_item_uid = {}
        self.object_uid_counter = 0
        self.player_idx_to_item_holding_uid = {}
        self.active_item_uids = []
        self.pot_states_tracking = {}


        self.player_idle_time = [0] * self.num_players

        self.set_layout_params(layout_name)

    def __eq__(self, other):
        return np.array_equal(self.terrain_mtx, other.terrain_mtx) and \
               self.start_player_positions == other.start_player_positions and \
               self.start_order_list == other.start_order_list and \
               self.soup_cooking_time == other.soup_cooking_time and \
               self.num_items_for_soup == other.num_items_for_soup and \
               self.delivery_reward == other.delivery_reward and \
               self.reward_shaping_params == other.reward_shaping_params and \
               self.layout_name == other.layout_name

    def copy(self):
        return OvercookedGridworld(
            terrain=self.terrain_mtx.copy(),
            start_player_positions=self.start_player_positions,
            start_order_list=None if self.start_order_list is None else list(self.start_order_list),
            cook_time=self.soup_cooking_time,
            num_items_for_soup=self.num_items_for_soup,
            delivery_reward=self.delivery_reward,
            rew_shaping_params=copy.deepcopy(self.reward_shaping_params),
            layout_name=self.layout_name
        )

    @property
    def mdp_params(self):
        return {
            "layout_name": self.layout_name,
            "terrain": self.terrain_mtx,
            "start_player_positions": self.start_player_positions,
            "start_order_list": self.start_order_list,
            "cook_time": self.soup_cooking_time,
            "num_items_for_soup": self.num_items_for_soup,
            "delivery_reward": self.delivery_reward,
            "rew_shaping_params": copy.deepcopy(self.reward_shaping_params)
        }

    @staticmethod
    def from_layout_name(layout_name, **params_to_overwrite):
        """
        Generates a OvercookedGridworld instance from a layout file.

        One can overwrite the default mdp configuration using partial_mdp_config.
        """
        params_to_overwrite = params_to_overwrite.copy()
        base_layout_params = read_layout_dict(layout_name)

        grid = base_layout_params['grid']
        del base_layout_params['grid']
        base_layout_params['layout_name'] = layout_name

        # Clean grid
        grid = [layout_row.strip() for layout_row in grid.split("\n")]
        return OvercookedGridworld.from_grid(grid, base_layout_params, params_to_overwrite)

    @staticmethod
    def from_grid(layout_grid, base_layout_params={}, params_to_overwrite={}, debug=False):
        """
        Returns instance of OvercookedGridworld with terrain and starting
        positions derived from layout_grid.
        One can override default configuration parameters of the mdp in
        partial_mdp_config.
        """
        mdp_config = base_layout_params.copy()

        layout_grid = [[c for c in row] for row in layout_grid]
        OvercookedGridworld._assert_valid_grid(layout_grid)

        player_positions = [None] * 9
        for y, row in enumerate(layout_grid):
            for x, c in enumerate(row):
                if c in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    layout_grid[y][x] = ' '

                    # -1 is to account for fact that player indexing starts from 1 rather than 0
                    assert player_positions[int(c) - 1] is None, 'Duplicate player in grid'
                    player_positions[int(c) - 1] = (x, y)

        num_players = len([x for x in player_positions if x is not None])
        player_positions = player_positions[:num_players]

        # After removing player positions from grid we have a terrain mtx
        mdp_config["terrain"] = layout_grid
        mdp_config["start_player_positions"] = player_positions

        for k, v in params_to_overwrite.items():
            curr_val = mdp_config[k]
            if debug:
                print("Overwriting mdp layout standard config value {}:{} -> {}".format(k, curr_val, v))
            mdp_config[k] = v

        return OvercookedGridworld(**mdp_config)

    def get_actions(self, state):
        """
        Returns the list of lists of valid actions for 'state'.

        The ith element of the list is the list of valid actions that player i
        can take.
        """
        self._check_valid_state(state)
        return [self._get_player_actions(state, i) for i in range(len(state.players))]

    def _get_player_actions(self, state, player_num):
        """All actions are allowed to all players in all states."""
        return Action.ALL_ACTIONS

    def _check_action(self, state, joint_action):
        for p_action, p_legal_actions in zip(joint_action, self.get_actions(state)):
            if p_action not in p_legal_actions:
                raise ValueError('Invalid action')

    def get_standard_start_state(self):
        start_state = OvercookedState.from_player_positions(
            self.start_player_positions, order_list=self.start_order_list
        )
        return start_state

    def get_random_start_state_fn(self, random_start_pos=False, rnd_obj_prob_thresh=0.0):
        def start_state_fn():
            if random_start_pos:
                valid_positions = self.get_valid_joint_player_positions()
                start_pos = valid_positions[np.random.choice(len(valid_positions))]
            else:
                start_pos = self.start_player_positions

            start_state = OvercookedState.from_player_positions(start_pos, order_list=self.start_order_list)

            if rnd_obj_prob_thresh == 0:
                return start_state

            # Arbitrary hard-coding for randomization of objects
            # For each pot, add a random amount of onions with prob rnd_obj_prob_thresh
            pots = self.get_pot_states(start_state)["empty"]
            for pot_loc in pots:
                p = np.random.rand()
                if p < rnd_obj_prob_thresh:
                    n = int(np.random.randint(low=1, high=4))
                    start_state.objects[pot_loc] = ObjectState("soup", pot_loc, ('onion', n, 0))

            # For each player, add a random object with prob rnd_obj_prob_thresh
            for player in start_state.players:
                p = np.random.rand()
                if p < rnd_obj_prob_thresh:
                    # Different objects have different probabilities
                    obj = np.random.choice(["dish", "onion", "soup"], p=[0.2, 0.6, 0.2])
                    if obj == "soup":
                        player.set_object(
                            ObjectState(obj, player.position,
                                        ('onion', self.num_items_for_soup, self.soup_cooking_time))
                        )
                    else:
                        player.set_object(ObjectState(obj, player.position))
            return start_state

        return start_state_fn

    def is_terminal(self, state):
        # There is a finite horizon, handled by the environment.
        if state.order_list is None:
            return False
        return len(state.order_list) == 0

    def get_valid_player_positions(self):
        return self.terrain_pos_dict[' ']

    def get_valid_joint_player_positions(self):
        """Returns all valid tuples of the form (p0_pos, p1_pos, p2_pos, ...)"""
        valid_positions = self.get_valid_player_positions()
        all_joint_positions = list(itertools.product(valid_positions, repeat=self.num_players))
        valid_joint_positions = [j_pos for j_pos in all_joint_positions if not self.is_joint_position_collision(j_pos)]
        return valid_joint_positions

    def get_valid_player_positions_and_orientations(self):
        valid_states = []
        for pos in self.get_valid_player_positions():
            valid_states.extend([(pos, d) for d in Direction.ALL_DIRECTIONS])
        return valid_states

    def get_valid_joint_player_positions_and_orientations(self):
        """All joint player position and orientation pairs that are not
        overlapping and on empty terrain."""
        valid_player_states = self.get_valid_player_positions_and_orientations()

        valid_joint_player_states = []
        for players_pos_and_orientations in itertools.product(valid_player_states, repeat=self.num_players):
            joint_position = [plyer_pos_and_or[0] for plyer_pos_and_or in players_pos_and_orientations]
            if not self.is_joint_position_collision(joint_position):
                valid_joint_player_states.append(players_pos_and_orientations)

        return valid_joint_player_states

    def get_adjacent_features(self, player):
        adj_feats = []
        pos = player.position
        for d in Direction.ALL_DIRECTIONS:
            adj_pos = Action.move_in_direction(pos, d)
            adj_feats.append((pos, self.get_terrain_type_at_pos(adj_pos)))
        return adj_feats

    def get_terrain_type_at_pos(self, pos):
        x, y = pos
        return self.terrain_mtx[y][x]

    def set_layout_params(self, layout_name):
        all_layout_params = {
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
            'unident_s': {
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
            'random1': {
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
            'simple': {
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
        for name in all_layout_params:
            if name != "random0":
                all_layout_params[name]['shared_counters'] = list(all_layout_params[name]['counter_location_to_id'].keys())
        self.layout_params_dict = all_layout_params[layout_name]
        return

    def get_layout_params(self, layout_name):
        return self.layout_params_dict[layout_name]

    def get_dish_dispenser_locations(self):
        return list(self.terrain_pos_dict['D'])

    def get_onion_dispenser_locations(self):
        return list(self.terrain_pos_dict['O'])

    def get_tomato_dispenser_locations(self):
        return list(self.terrain_pos_dict['T'])

    def get_serving_locations(self):
        return list(self.terrain_pos_dict['S'])

    def get_pot_locations(self):
        return list(self.terrain_pos_dict['P'])

    def get_counter_locations(self):
        return list(self.terrain_pos_dict['X'])

    def get_pot_states(self, state):
        """Returns dict with structure:
        {
         empty: [ObjStates]
         onion: {
            'x_items': [soup objects with x items],
            'cooking': [ready soup objs]
            'ready': [ready soup objs],
            'partially_full': [all non-empty and non-full soups]
            }
         tomato: same dict structure as above
        }
        """
        pots_states_dict = {}
        pots_states_dict['empty'] = []
        pots_states_dict['onion'] = defaultdict(list)
        pots_states_dict['tomato'] = defaultdict(list)
        for pot_pos in self.get_pot_locations():
            if not state.has_object(pot_pos):
                pots_states_dict['empty'].append(pot_pos)
            else:
                soup_obj = state.get_object(pot_pos)
                soup_type, num_items, cook_time = soup_obj.state
                if 0 < num_items < self.num_items_for_soup:
                    pots_states_dict[soup_type]['{}_items'.format(num_items)].append(pot_pos)
                elif num_items == self.num_items_for_soup:
                    # print("cook time, self.soup cooking time", (cook_time, self.soup_cooking_time))
                    self.soup_cooking_time = 20
                    assert cook_time <= self.soup_cooking_time
                    if cook_time == self.soup_cooking_time:
                        pots_states_dict[soup_type]['ready'].append(pot_pos)
                    else:
                        pots_states_dict[soup_type]['cooking'].append(pot_pos)
                else:
                    raise ValueError("Pot with more than {} items".format(self.num_items_for_soup))

                if 0 < num_items < self.num_items_for_soup:
                    pots_states_dict[soup_type]['partially_full'].append(pot_pos)

        return pots_states_dict

    def get_counter_objects_dict(self, state, counter_subset=None):
        """Returns a dictionary of pos:objects on counters by type"""
        counters_considered = self.terrain_pos_dict['X'] if counter_subset is None else counter_subset
        counter_objects_dict = defaultdict(list)
        for obj in state.objects.values():
            if obj.position in counters_considered:
                counter_objects_dict[obj.name].append(obj.position)
        return counter_objects_dict

    def get_empty_counter_locations(self, state):
        counter_locations = self.get_counter_locations()
        return [pos for pos in counter_locations if not state.has_object(pos)]

    def get_state_transition(self, state, joint_action):
        """Gets information about possible transitions for the action.

        Returns the next state, sparse reward and reward shaping.
        Assumes all actions are deterministic.

        NOTE: Sparse reward is given only when soups are delivered,
        shaped reward is given only for completion of subgoals
        (not soup deliveries).
        """
        assert not self.is_terminal(state), "Trying to find successor of a terminal state: {}".format(state)
        for action, action_set in zip(joint_action, self.get_actions(state)):
            if action not in action_set:
                raise ValueError("Illegal action %s in state %s" % (action, state))

        new_state = state.deepcopy()

        # Resolve interacts first
        sparse_reward, shaped_reward = self.resolve_interacts(new_state, joint_action)

        assert new_state.player_positions == state.player_positions
        assert new_state.player_orientations == state.player_orientations

        # Resolve player movements
        collided = self.resolve_movement(new_state, joint_action)
        COLLISION_PENALTY = -1
        if collided:
            shaped_reward += COLLISION_PENALTY

        # Finally, environment effects
        sparse_reward += self.step_environment_effects(new_state)

        # Additional dense reward logic
        # shaped_reward += self.calculate_distance_based_shaped_reward(state, new_state)

        return new_state, sparse_reward, shaped_reward

    def resolve_interacts_old(self, new_state, joint_action):
        """
        Resolve any INTERACT actions, if present.

        Currently if two players both interact with a terrain, we resolve player 1's interact
        first and then player 2's, without doing anything like collision checking.
        """
        pot_states = self.get_pot_states(new_state)
        ready_pots = pot_states["tomato"]["ready"] + pot_states["onion"]["ready"]
        cooking_pots = ready_pots + pot_states["tomato"]["cooking"] + pot_states["onion"]["cooking"]
        nearly_ready_pots = cooking_pots + pot_states["tomato"]["partially_full"] + pot_states["onion"][
            "partially_full"]
        full_pots = cooking_pots + ready_pots
        num_pots = len(pot_states)

        # self.mean_limbo_time = max([self.item_tracking_dict[item_uid]['limbo_time'] for item_uid in self.item_tracking_dict]) \
        #     if len(self.item_tracking_dict) > 0 else 0

        # self.mean_handoff_time = max(
        #     [self.item_tracking_dict[item_uid]['handoff_time'] for item_uid in self.item_tracking_dict]) \
        #     if len(self.item_tracking_dict) > 0 else 0

        handoff_times_list = []
        for item_uid in self.item_tracking_dict:
            if self.item_tracking_dict[item_uid]['handoff_time'] > 0:
                handoff_times_list.append(self.item_tracking_dict[item_uid]['handoff_time'])

        self.mean_handoff_time = np.mean(handoff_times_list)
        # self.mean_handoff_time = np.mean(handoff_times_list) + np.std(handoff_times_list)
        # if self.mean_handoff_time < 3:
        #     self.mean_handoff_time = 100000000

        for item_uid in self.active_item_uids:
            self.item_tracking_dict[item_uid]['limbo_time'] += 1
            if self.item_tracking_dict[item_uid]['player_holding'] == None:
                self.item_tracking_dict[item_uid]['handoff_time'] += 1

        # HANDOFF_TIME_REWARD = 1
        # HANDOFF_TIME_OVER_MEAN_PENALTY = -2
        # FC_MEAN_LIMBO = 74
        # FC_MEAN_TRANSFER = 63
        #
        # OVERPASSING_PENALTY = -2
        # USE_BOTH_POTS_REWARD = 2
        # HIGH_POT_COVERAGE_SIMUL_COOKING_REWARD = 0
        # NUM_TOUCHES_PENALTY = -2


        HANDOFF_TIME_REWARD = 0
        HANDOFF_TIME_OVER_MEAN_PENALTY = 0
        FC_MEAN_LIMBO = 74
        FC_MEAN_TRANSFER = 63

        OVERPASSING_PENALTY = 0
        USE_BOTH_POTS_REWARD = 0
        HIGH_POT_COVERAGE_SIMUL_COOKING_REWARD = 0



        sparse_reward, shaped_reward = 0, 0
        player_idx = -1
        for player, action in zip(new_state.players, joint_action):
            player_idx += 1
            # if action == Action.STAY:
            #     # if player.position[0] == 1:
            #     #     player_idx = 0
            #     # else:
            #     #     player_idx = 1
            #     self.player_idle_time[player_idx] += 1
            #
            # if 'IDLE_DIFF_REWARD' in self.reward_shaping_params:
            #     if np.var(self.player_idle_time) < 5:
            #         shaped_reward += self.reward_shaping_params["IDLE_DIFF_REWARD"]
            #     else:
            #         shaped_reward -= self.reward_shaping_params["IDLE_DIFF_REWARD"]
            #
            if action != Action.INTERACT:
                continue


            pos, o = player.position, player.orientation
            i_pos = Action.move_in_direction(pos, o)
            terrain_type = self.get_terrain_type_at_pos(i_pos)

            if terrain_type == 'X':

                if player.has_object() and not new_state.has_object(i_pos):
                    # Action Type 1: Player put object on counter
                    new_state.add_object(player.remove_object(), i_pos)
                    # **ADDED **** Player put an object down on the counter
                    # if full_pots == num_pots:
                    #     shaped_reward += 0.1*self.reward_shaping_params["DISH_PICKUP_REWARD"]

                    # If player placed an obj on counter
                    # if player.get_object().name in ['onion', 'tomato', 'dish']:
                    try:
                        placed_item_uid = self.player_idx_to_item_holding_uid[player_idx]

                        self.item_tracking_dict[placed_item_uid]['player_holding'] = None
                        self.item_tracking_dict[placed_item_uid]['location_placed'] = i_pos

                        self.location_to_item_uid[i_pos] = placed_item_uid
                        self.item_uid_to_location[placed_item_uid] = i_pos
                        self.player_idx_to_item_holding_uid[player_idx] = None
                    except:
                        pass



                elif not player.has_object() and new_state.has_object(i_pos):
                    # Action Type 2: Player picked object up from counter
                    player.set_object(new_state.remove_object(i_pos))

                    try:
                        picked_item_uid = self.location_to_item_uid[i_pos]

                        self.item_tracking_dict[picked_item_uid]['player_holding'] = player_idx
                        self.item_tracking_dict[picked_item_uid]['location_placed'] = None
                        self.item_tracking_dict[picked_item_uid]['past_players'].append(player_idx)
                        self.item_tracking_dict[picked_item_uid]['n_touches'] += 1

                        if len(np.unique(self.item_tracking_dict[picked_item_uid]['past_players'])) > 1 \
                                and self.item_tracking_dict[picked_item_uid]['handoff_time'] <= self.mean_handoff_time:
                            shaped_reward += HANDOFF_TIME_REWARD
                        # if len(np.unique(self.item_tracking_dict[picked_item_uid]['past_players'])) > 1 \
                        #         and self.item_tracking_dict[picked_item_uid]['handoff_time'] > self.mean_handoff_time:
                        #     if self.mean_handoff_time > 3:
                        #         shaped_reward -= HANDOFF_TIME_REWARD
                        # if len(np.unique(self.item_tracking_dict[picked_item_uid]['past_players'])) > 1 \
                        #         and self.item_tracking_dict[picked_item_uid]['handoff_time'] > FC_MEAN_TRANSFER:
                        #     shaped_reward += HANDOFF_TIME_OVER_MEAN_PENALTY

                        if self.item_tracking_dict[picked_item_uid]['n_touches'] > len(np.unique(self.item_tracking_dict[picked_item_uid]['past_players'])):
                            shaped_reward += NUM_TOUCHES_PENALTY

                        # if self.item_tracking_dict[picked_item_uid]['n_touches'] > 4:
                        #     shaped_reward += OVERPASSING_PENALTY


                        self.location_to_item_uid[i_pos] = None
                        self.item_uid_to_location[picked_item_uid] = None
                        self.player_idx_to_item_holding_uid[player_idx] = picked_item_uid
                    except:
                        pass


            elif terrain_type == 'O' and player.held_object is None:
                # Action Type 3: Player picked up onion from dispenser

                player.set_object(ObjectState('onion', pos))
                # **ADDED **** Player picked up an onion
                # If there were two full pots
                # if full_pots == num_pots:
                #     shaped_reward -= self.reward_shaping_params["DISH_PICKUP_REWARD"]

                # If player picked up an onion from dispenser
                try:
                    self.object_uid_counter += 1
                    new_obj_record = {
                        'player_holding': player_idx,
                        'past_players': [player_idx],
                        'location_placed': None,
                        'name': 'onion',
                        'limbo_time': 0,
                        'handoff_time': 0,
                        'completed': False,
                        'n_touches': 1
                    }
                    self.item_tracking_dict[self.object_uid_counter] = new_obj_record
                    self.player_idx_to_item_holding_uid[player_idx] = self.object_uid_counter
                    self.active_item_uids.append(self.object_uid_counter)
                except:
                    pass


            elif terrain_type == 'T' and player.held_object is None:
                # Action Type 4: Player picked up tomato from dispenser
                player.set_object(ObjectState('tomato', pos))

                # If player picked up an tomato from dispenser
                self.object_uid_counter += 1
                new_obj_record = {
                    'player_holding': player_idx,
                    'past_players': [player_idx],
                    'location_placed': None,
                    'name': 'tomato',
                    'limbo_time': 0,
                    'handoff_time': 0,
                    'completed': False,
                    'n_touches': 1
                }
                self.item_tracking_dict[self.object_uid_counter] = new_obj_record
                self.player_idx_to_item_holding_uid[player_idx] = self.object_uid_counter
                self.active_item_uids.append(self.object_uid_counter)


            elif terrain_type == 'D' and player.held_object is None:
                # Action Type 5: Player picked up dish from dispenser
                dishes_already = len(new_state.player_objects_by_type['dish'])
                player.set_object(ObjectState('dish', pos))

                dishes_on_counters = self.get_counter_objects_dict(new_state)["dish"]
                if len(nearly_ready_pots) > dishes_already and len(dishes_on_counters) == 0:
                    shaped_reward += self.reward_shaping_params["DISH_PICKUP_REWARD"]

                # If player picked up an dish from dispenser
                self.object_uid_counter += 1
                new_obj_record = {
                    'player_holding': player_idx,
                    'past_players': [player_idx],
                    'location_placed': None,
                    'name': 'dish',
                    'limbo_time': 0,
                    'handoff_time': 0,
                    'completed': False,
                    'n_touches': 1
                }
                self.item_tracking_dict[self.object_uid_counter] = new_obj_record
                self.player_idx_to_item_holding_uid[player_idx] = self.object_uid_counter
                self.active_item_uids.append(self.object_uid_counter)

            elif terrain_type == 'P' and player.has_object():
                if player.get_object().name == 'dish' and new_state.has_object(i_pos):
                    # Action Type 6: Player picked up soup from pot with dish
                    obj = new_state.get_object(i_pos)
                    assert obj.name == 'soup', 'Object in pot was not soup'
                    _, num_items, cook_time = obj.state
                    if num_items == self.num_items_for_soup and cook_time >= self.soup_cooking_time:
                        player.remove_object()  # Turn the dish into the soup
                        player.set_object(new_state.remove_object(i_pos))
                        shaped_reward += self.reward_shaping_params["SOUP_PICKUP_REWARD"]

                elif player.get_object().name in ['onion', 'tomato']:
                    item_type = player.get_object().name

                    if not new_state.has_object(i_pos):
                        # Action Type 7: Player placed onion or tomato in empty pot
                        # Pot was empty
                        player.remove_object()
                        new_state.add_object(ObjectState('soup', i_pos, (item_type, 1, 0)), i_pos)
                        shaped_reward += self.reward_shaping_params["PLACEMENT_IN_POT_REW"]

                        # Onion placed in pot is no longer active
                        try:
                            placed_item_uid = self.player_idx_to_item_holding_uid[player_idx]

                            self.item_tracking_dict[placed_item_uid]['player_holding'] = None
                            self.item_tracking_dict[placed_item_uid]['location_placed'] = i_pos
                            self.item_tracking_dict[placed_item_uid]['completed'] = True

                            self.player_idx_to_item_holding_uid[player_idx] = None

                            self.active_item_uids.remove(placed_item_uid)
                        except:
                            pass


                    else:
                        # Action Type 8: Player placed onion in partially filled pot
                        # Pot has already items in it
                        obj = new_state.get_object(i_pos)
                        assert obj.name == 'soup', 'Object in pot was not soup'
                        soup_type, num_items, cook_time = obj.state
                        if num_items < self.num_items_for_soup and soup_type == item_type:
                            player.remove_object()
                            obj.state = (soup_type, num_items + 1, 0)
                            shaped_reward += self.reward_shaping_params["PLACEMENT_IN_POT_REW"]

                            current_pots_states_dict = self.get_pot_states(new_state)
                            num_cooking_pots = len(current_pots_states_dict['onion']['cooking'])
                            if num_cooking_pots > 1:
                                shaped_reward += HIGH_POT_COVERAGE_SIMUL_COOKING_REWARD

                            # Onion placed in pot is no longer active
                            try:
                                placed_item_uid = self.player_idx_to_item_holding_uid[player_idx]

                                self.item_tracking_dict[placed_item_uid]['player_holding'] = None
                                self.item_tracking_dict[placed_item_uid]['location_placed'] = i_pos
                                self.item_tracking_dict[placed_item_uid]['completed'] = True

                                self.player_idx_to_item_holding_uid[player_idx] = None

                                self.active_item_uids.remove(placed_item_uid)
                            except:
                                pass

            elif terrain_type == 'S' and player.has_object():
                obj = player.get_object()
                if obj.name == 'soup':
                    # Action Type 9: Player delivered soup
                    new_state, delivery_rew = self.deliver_soup(new_state, player, obj)
                    sparse_reward += delivery_rew

                    # Dish served is no longer active
                    try:
                        placed_item_uid = self.player_idx_to_item_holding_uid[player_idx]

                        self.item_tracking_dict[placed_item_uid]['player_holding'] = None
                        self.item_tracking_dict[placed_item_uid]['location_placed'] = None
                        self.item_tracking_dict[placed_item_uid]['completed'] = True

                        self.player_idx_to_item_holding_uid[player_idx] = None

                        self.active_item_uids.remove(placed_item_uid)
                    except:
                        pass

                    # If last soup necessary was delivered, stop resolving interacts
                    if new_state.order_list is not None and len(new_state.order_list) == 0:
                        break

        return sparse_reward, shaped_reward

    def resolve_interacts(self, new_state, joint_action):
        """
        Resolve any INTERACT actions, if present.

        Currently if two players both interact with a terrain, we resolve player 1's interact
        first and then player 2's, without doing anything like collision checking.

        State, Action Featurization
        1. Onion placed in empty pot
        2. Onion placed in partially full pot
        3. Dish picked up from dispenser if no dishes on counters, and # nearly ready pots > dishes out already
        4. Soup picked up from ready pot
        5. Both pots cooking simultaneously
        8/6. Serve soup
        7. Shared counter usage

        """
        pot_states = self.get_pot_states(new_state)
        ready_pots = pot_states["tomato"]["ready"] + pot_states["onion"]["ready"]
        cooking_pots = ready_pots + pot_states["tomato"]["cooking"] + pot_states["onion"]["cooking"]
        nearly_ready_pots = cooking_pots + pot_states["tomato"]["partially_full"] + pot_states["onion"][
            "partially_full"]
        full_pots = cooking_pots + ready_pots
        num_pots = len(pot_states)

        # self.mean_limbo_time = max([self.item_tracking_dict[item_uid]['limbo_time'] for item_uid in self.item_tracking_dict]) \
        #     if len(self.item_tracking_dict) > 0 else 0

        # self.mean_handoff_time = max(
        #     [self.item_tracking_dict[item_uid]['handoff_time'] for item_uid in self.item_tracking_dict]) \
        #     if len(self.item_tracking_dict) > 0 else 0

        handoff_times_list = []
        for item_uid in self.item_tracking_dict:
            if self.item_tracking_dict[item_uid]['handoff_time'] > 0:
                handoff_times_list.append(self.item_tracking_dict[item_uid]['handoff_time'])

        self.mean_handoff_time = np.mean(handoff_times_list)
        # self.mean_handoff_time = np.mean(handoff_times_list) + np.std(handoff_times_list)
        # if self.mean_handoff_time < 3:
        #     self.mean_handoff_time = 100000000

        for item_uid in self.active_item_uids:
            self.item_tracking_dict[item_uid]['limbo_time'] += 1
            if self.item_tracking_dict[item_uid]['player_holding'] == None:
                self.item_tracking_dict[item_uid]['handoff_time'] += 1

        # HANDOFF_TIME_REWARD = 1
        # HANDOFF_TIME_OVER_MEAN_PENALTY = -2
        # FC_MEAN_LIMBO = 74
        # FC_MEAN_TRANSFER = 63
        #
        # OVERPASSING_PENALTY = -2
        # USE_BOTH_POTS_REWARD = 2
        # HIGH_POT_COVERAGE_SIMUL_COOKING_REWARD = 0
        # NUM_TOUCHES_PENALTY = -4


        HANDOFF_TIME_REWARD = 1
        HANDOFF_TIME_OVER_MEAN_PENALTY = 0
        FC_MEAN_LIMBO = 74
        FC_MEAN_TRANSFER = 63

        OVERPASSING_PENALTY = 0
        USE_BOTH_POTS_REWARD = 0
        HIGH_POT_COVERAGE_SIMUL_COOKING_REWARD = 0

        shared_counters = [(2, 1), (2, 2), (2, 3)]


        sparse_reward, shaped_reward = 0, 0
        player_idx = -1
        for player, action in zip(new_state.players, joint_action):
            player_idx += 1
            # if action == Action.STAY:
            #     # if player.position[0] == 1:
            #     #     player_idx = 0
            #     # else:
            #     #     player_idx = 1
            #     self.player_idle_time[player_idx] += 1
            #
            # if 'IDLE_DIFF_REWARD' in self.reward_shaping_params:
            #     if np.var(self.player_idle_time) < 5:
            #         shaped_reward += self.reward_shaping_params["IDLE_DIFF_REWARD"]
            #     else:
            #         shaped_reward -= self.reward_shaping_params["IDLE_DIFF_REWARD"]
            #
            if action != Action.INTERACT:
                continue


            pos, o = player.position, player.orientation
            i_pos = Action.move_in_direction(pos, o)
            terrain_type = self.get_terrain_type_at_pos(i_pos)

            if terrain_type == 'X':

                if player.has_object() and not new_state.has_object(i_pos):
                    # Action Type 1: Player put object on counter
                    new_state.add_object(player.remove_object(), i_pos)
                    # **ADDED **** Player put an object down on the counter
                    # if full_pots == num_pots:
                    #     shaped_reward += 0.1*self.reward_shaping_params["DISH_PICKUP_REWARD"]

                    # If player placed an obj on counter
                    # if player.get_object().name in ['onion', 'tomato', 'dish']:
                    try:
                        placed_item_uid = self.player_idx_to_item_holding_uid[player_idx]

                        self.item_tracking_dict[placed_item_uid]['player_holding'] = None
                        self.item_tracking_dict[placed_item_uid]['location_placed'] = i_pos

                        self.location_to_item_uid[i_pos] = placed_item_uid
                        self.item_uid_to_location[placed_item_uid] = i_pos
                        self.player_idx_to_item_holding_uid[player_idx] = None
                    except:
                        pass



                elif not player.has_object() and new_state.has_object(i_pos):
                    # Action Type 2: Player picked object up from counter
                    player.set_object(new_state.remove_object(i_pos))

                    if len(self.reward_shaping_params)==7:
                        if tuple(i_pos) in shared_counters:
                            shaped_reward += (self.reward_shaping_params['SHARED_COUNTER_REWARD'])

                    try:
                        picked_item_uid = self.location_to_item_uid[i_pos]

                        self.item_tracking_dict[picked_item_uid]['player_holding'] = player_idx
                        self.item_tracking_dict[picked_item_uid]['location_placed'] = None
                        self.item_tracking_dict[picked_item_uid]['past_players'].append(player_idx)
                        self.item_tracking_dict[picked_item_uid]['n_touches'] += 1

                        if len(np.unique(self.item_tracking_dict[picked_item_uid]['past_players'])) > 1 \
                                and len(np.unique(self.item_tracking_dict[picked_item_uid]['past_players'])) < 3:
                            if self.item_tracking_dict[picked_item_uid]['handoff_time'] <= self.mean_handoff_time:
                                shaped_reward += HANDOFF_TIME_REWARD
                        # if len(np.unique(self.item_tracking_dict[picked_item_uid]['past_players'])) > 1 \
                        #         and self.item_tracking_dict[picked_item_uid]['handoff_time'] > self.mean_handoff_time:
                        #     if self.mean_handoff_time > 3:
                        #         shaped_reward -= HANDOFF_TIME_REWARD
                        # if len(np.unique(self.item_tracking_dict[picked_item_uid]['past_players'])) > 1 \
                        #         and self.item_tracking_dict[picked_item_uid]['handoff_time'] > FC_MEAN_TRANSFER:
                        #     shaped_reward += HANDOFF_TIME_OVER_MEAN_PENALTY

                        # if self.item_tracking_dict[picked_item_uid]['n_touches'] > len(np.unique(self.item_tracking_dict[picked_item_uid]['past_players'])):
                        #     shaped_reward += NUM_TOUCHES_PENALTY

                        # if self.item_tracking_dict[picked_item_uid]['n_touches'] > 4:
                        #     shaped_reward += OVERPASSING_PENALTY


                        self.location_to_item_uid[i_pos] = None
                        self.item_uid_to_location[picked_item_uid] = None
                        self.player_idx_to_item_holding_uid[player_idx] = picked_item_uid
                    except:
                        pass


            elif terrain_type == 'O' and player.held_object is None:
                # Action Type 3: Player picked up onion from dispenser

                player.set_object(ObjectState('onion', pos))
                # **ADDED **** Player picked up an onion
                # If there were two full pots
                # if full_pots == num_pots:
                #     shaped_reward -= self.reward_shaping_params["DISH_PICKUP_REWARD"]

                # If player picked up an onion from dispenser
                try:
                    self.object_uid_counter += 1
                    new_obj_record = {
                        'player_holding': player_idx,
                        'past_players': [player_idx],
                        'location_placed': None,
                        'name': 'onion',
                        'limbo_time': 0,
                        'handoff_time': 0,
                        'completed': False,
                        'n_touches': 1
                    }
                    self.item_tracking_dict[self.object_uid_counter] = new_obj_record
                    self.player_idx_to_item_holding_uid[player_idx] = self.object_uid_counter
                    self.active_item_uids.append(self.object_uid_counter)
                except:
                    pass


            elif terrain_type == 'T' and player.held_object is None:
                # Action Type 4: Player picked up tomato from dispenser
                player.set_object(ObjectState('tomato', pos))

                # If player picked up an tomato from dispenser
                self.object_uid_counter += 1
                new_obj_record = {
                    'player_holding': player_idx,
                    'past_players': [player_idx],
                    'location_placed': None,
                    'name': 'tomato',
                    'limbo_time': 0,
                    'handoff_time': 0,
                    'completed': False,
                    'n_touches': 1
                }
                self.item_tracking_dict[self.object_uid_counter] = new_obj_record
                self.player_idx_to_item_holding_uid[player_idx] = self.object_uid_counter
                self.active_item_uids.append(self.object_uid_counter)


            elif terrain_type == 'D' and player.held_object is None:
                # Action Type 5: Player picked up dish from dispenser
                dishes_already = len(new_state.player_objects_by_type['dish'])
                player.set_object(ObjectState('dish', pos))

                dishes_on_counters = self.get_counter_objects_dict(new_state)["dish"]
                if len(nearly_ready_pots) > dishes_already and len(dishes_on_counters) == 0:
                    shaped_reward += self.reward_shaping_params["DISH_PICKUP_REWARD"]

                if (4, 1) in full_pots and (3, 0) in full_pots:
                    shaped_reward += self.reward_shaping_params["BOTH_POTS_FULL_REWARD"]

                # If player picked up an dish from dispenser
                self.object_uid_counter += 1
                new_obj_record = {
                    'player_holding': player_idx,
                    'past_players': [player_idx],
                    'location_placed': None,
                    'name': 'dish',
                    'limbo_time': 0,
                    'handoff_time': 0,
                    'completed': False,
                    'n_touches': 1
                }
                self.item_tracking_dict[self.object_uid_counter] = new_obj_record
                self.player_idx_to_item_holding_uid[player_idx] = self.object_uid_counter
                self.active_item_uids.append(self.object_uid_counter)

            elif terrain_type == 'P' and player.has_object():
                if player.get_object().name == 'dish' and new_state.has_object(i_pos):
                    # Action Type 6: Player picked up soup from pot with dish
                    obj = new_state.get_object(i_pos)
                    assert obj.name == 'soup', 'Object in pot was not soup'
                    _, num_items, cook_time = obj.state
                    if num_items == self.num_items_for_soup and cook_time >= self.soup_cooking_time:
                        player.remove_object()  # Turn the dish into the soup
                        player.set_object(new_state.remove_object(i_pos))
                        shaped_reward += self.reward_shaping_params["SOUP_PICKUP_FROM_READY_POT_REWARD"]

                elif player.get_object().name in ['onion', 'tomato']:
                    item_type = player.get_object().name

                    if not new_state.has_object(i_pos):
                        # Action Type 7: Player placed onion or tomato in empty pot
                        # Pot was empty
                        player.remove_object()
                        new_state.add_object(ObjectState('soup', i_pos, (item_type, 1, 0)), i_pos)
                        shaped_reward += self.reward_shaping_params["ONION_IN_EMPTY_POT_REWARD"]

                        # Onion placed in pot is no longer active
                        try:
                            placed_item_uid = self.player_idx_to_item_holding_uid[player_idx]

                            self.item_tracking_dict[placed_item_uid]['player_holding'] = None
                            self.item_tracking_dict[placed_item_uid]['location_placed'] = i_pos
                            self.item_tracking_dict[placed_item_uid]['completed'] = True

                            self.player_idx_to_item_holding_uid[player_idx] = None

                            self.active_item_uids.remove(placed_item_uid)
                        except:
                            pass


                    else:
                        # Action Type 8: Player placed onion in partially filled pot
                        # Pot has already items in it
                        obj = new_state.get_object(i_pos)
                        assert obj.name == 'soup', 'Object in pot was not soup'
                        soup_type, num_items, cook_time = obj.state
                        if num_items < self.num_items_for_soup and soup_type == item_type:
                            player.remove_object()
                            obj.state = (soup_type, num_items + 1, 0)
                            shaped_reward += self.reward_shaping_params["ONION_IN_PARTIAL_POT_REWARD"]




                            current_pots_states_dict = self.get_pot_states(new_state)
                            num_cooking_pots = len(current_pots_states_dict['onion']['cooking'])
                            # if num_cooking_pots > 1:
                            #     shaped_reward += HIGH_POT_COVERAGE_SIMUL_COOKING_REWARD
                            ready_pots = pot_states["onion"]["ready"]
                            cooking_pots = pot_states["onion"]["cooking"]
                            full_pots = cooking_pots + ready_pots

                            #### IF both pots full
                            if (4, 1) in full_pots and (3, 0) in full_pots:
                                shaped_reward += self.reward_shaping_params["BOTH_POTS_FULL_REWARD"]

                            # Onion placed in pot is no longer active
                            try:
                                placed_item_uid = self.player_idx_to_item_holding_uid[player_idx]

                                self.item_tracking_dict[placed_item_uid]['player_holding'] = None
                                self.item_tracking_dict[placed_item_uid]['location_placed'] = i_pos
                                self.item_tracking_dict[placed_item_uid]['completed'] = True

                                self.player_idx_to_item_holding_uid[player_idx] = None

                                self.active_item_uids.remove(placed_item_uid)
                            except:
                                pass

            elif terrain_type == 'S' and player.has_object():
                obj = player.get_object()
                if obj.name == 'soup':
                    # Action Type 9: Player delivered soup
                    new_state, delivery_rew = self.deliver_soup(new_state, player, obj)
                    sparse_reward += delivery_rew
                    shaped_reward += self.reward_shaping_params["SERVE_SOUP_REWARD"]

                    # Dish served is no longer active
                    try:
                        placed_item_uid = self.player_idx_to_item_holding_uid[player_idx]

                        self.item_tracking_dict[placed_item_uid]['player_holding'] = None
                        self.item_tracking_dict[placed_item_uid]['location_placed'] = None
                        self.item_tracking_dict[placed_item_uid]['completed'] = True

                        self.player_idx_to_item_holding_uid[player_idx] = None

                        self.active_item_uids.remove(placed_item_uid)
                    except:
                        pass

                    # If last soup necessary was delivered, stop resolving interacts
                    if new_state.order_list is not None and len(new_state.order_list) == 0:
                        break

        return sparse_reward, shaped_reward

    def get_high_level_interact_action(self, new_state, joint_action, n_features=7):
        """
        Resolve any INTERACT actions, if present.

        Currently if two players both interact with a terrain, we resolve player 1's interact
        first and then player 2's, without doing anything like collision checking.


        State, Action Featurization
        1. Onion placed in empty pot
        2. Onion placed in partially full pot
        3. Dish picked up from dispenser if no dishes on counters, and # nearly ready pots > dishes out already
        4. Soup picked up from ready pot
        5. Both pots cooking simultaneously
        8/6. Serve soup
        7. Shared counter usage

        6. Handoff time of object picked up <= self mean
        7. Number of touches on onion or dish == len unique players

        """
        pot_states = self.get_pot_states(new_state)
        ready_pots = pot_states["tomato"]["ready"] + pot_states["onion"]["ready"]
        cooking_pots = ready_pots + pot_states["tomato"]["cooking"] + pot_states["onion"]["cooking"]
        nearly_ready_pots = cooking_pots + pot_states["tomato"]["partially_full"] + pot_states["onion"][
            "partially_full"]
        full_pots = cooking_pots + ready_pots
        num_pots = len(pot_states)

        shared_counters = self.layout_params_dict['shared_counters']
        pot_locations = self.layout_params_dict['pot_locations']

        reward_featurized_state = [0]*n_features

        # if (4,1) in full_pots and (3,0) in full_pots:
        #     # print("NUM FULL POTS = 2", full_pots)
        #     reward_featurized_state[4] = 0.15
        if len(pot_locations)==2:
            if pot_locations[0] in full_pots and pot_locations[1] in full_pots:
                reward_featurized_state[4] = 0.15
        # elif len(pot_locations)==1:
        #     if pot_locations[0] in full_pots:
        #         reward_featurized_state[4] = 0

        # self.mean_limbo_time = max([self.item_tracking_dict[item_uid]['limbo_time'] for item_uid in self.item_tracking_dict]) \
        #     if len(self.item_tracking_dict) > 0 else 0

        # self.mean_handoff_time = max(
        #     [self.item_tracking_dict[item_uid]['handoff_time'] for item_uid in self.item_tracking_dict]) \
        #     if len(self.item_tracking_dict) > 0 else 0

        handoff_times_list = []
        for item_uid in self.item_tracking_dict:
            if self.item_tracking_dict[item_uid]['handoff_time'] > 0:
                handoff_times_list.append(self.item_tracking_dict[item_uid]['handoff_time'])

        self.mean_handoff_time = np.mean(handoff_times_list)
        # self.mean_handoff_time = np.mean(handoff_times_list) + np.std(handoff_times_list)
        # if self.mean_handoff_time < 3:
        #     self.mean_handoff_time = 100000000

        for item_uid in self.active_item_uids:
            self.item_tracking_dict[item_uid]['limbo_time'] += 1
            if self.item_tracking_dict[item_uid]['player_holding'] == None:
                self.item_tracking_dict[item_uid]['handoff_time'] += 1

        # HANDOFF_TIME_REWARD = 1
        # HANDOFF_TIME_OVER_MEAN_PENALTY = -2
        # FC_MEAN_LIMBO = 74
        # FC_MEAN_TRANSFER = 63
        #
        # OVERPASSING_PENALTY = -2
        # USE_BOTH_POTS_REWARD = 2
        # HIGH_POT_COVERAGE_SIMUL_COOKING_REWARD = 0
        # NUM_TOUCHES_PENALTY = -2


        HANDOFF_TIME_REWARD = 0
        HANDOFF_TIME_OVER_MEAN_PENALTY = 0
        FC_MEAN_LIMBO = 74
        FC_MEAN_TRANSFER = 63

        OVERPASSING_PENALTY = 0
        USE_BOTH_POTS_REWARD = 0
        HIGH_POT_COVERAGE_SIMUL_COOKING_REWARD = 0

        # HIGH LEVEL ACTIONS LIST
        NORTH = 0
        SOUTH = 1
        EAST = 3
        WEST = 4
        STAY = 5
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
        ALL_HIGH_LEVEL_ACTIONS = [NORTH, SOUTH, EAST, WEST, STAY, PICKUP_ONION_FROM_DISPENSER,
                                  PICKUP_DISH_FROM_DISPENSER,
                                  PUT_DOWN_ONION_ON_COUNTER, PUT_DOWN_ONION_IN_POT, PUT_DOWN_DISH_ON_COUNTER, PUT_DOWN_SOUP_ON_COUNTER,
                                  PICKUP_ONION_FROM_COUNTER, PICKUP_DISH_FROM_COUNTER, PICKUP_SOUP_FROM_COUNTER,
                                  PICKUP_SOUP_W_DISH, SERVE_SOUP]

        player_idx_to_high_level_action = {0: STAY, 1:STAY}

        sparse_reward, shaped_reward = 0, 0
        player_idx = -1
        for player, action in zip(new_state.players, joint_action):
            player_idx += 1
            # if action == Action.STAY:
            #     # if player.position[0] == 1:
            #     #     player_idx = 0
            #     # else:
            #     #     player_idx = 1
            #     self.player_idle_time[player_idx] += 1
            #
            # if 'IDLE_DIFF_REWARD' in self.reward_shaping_params:
            #     if np.var(self.player_idle_time) < 5:
            #         shaped_reward += self.reward_shaping_params["IDLE_DIFF_REWARD"]
            #     else:
            #         shaped_reward -= self.reward_shaping_params["IDLE_DIFF_REWARD"]
            #
            if action != Action.INTERACT:
                if action == Action.STAY:
                    player_idx_to_high_level_action[player_idx] = STAY
                else:
                    player_idx_to_high_level_action[player_idx] = ACTION_DIRECTION_TO_HIGH_LEVEL[action]
                continue


            pos, o = player.position, player.orientation
            i_pos = Action.move_in_direction(pos, o)
            terrain_type = self.get_terrain_type_at_pos(i_pos)

            if terrain_type == 'X':

                if player.has_object() and not new_state.has_object(i_pos):
                    # Action Type 1: Player put object on counter

                    # **ADDED **** Player put an object down on the counter
                    # if full_pots == num_pots:
                    #     shaped_reward += 0.1*self.reward_shaping_params["DISH_PICKUP_REWARD"]

                    # If player placed an obj on counter
                    # if player.get_object().name in ['onion', 'tomato', 'dish']:
                    if player.get_object().name == 'onion':
                        player_idx_to_high_level_action[player_idx] = PUT_DOWN_ONION_ON_COUNTER
                    if player.get_object().name == 'dish':
                        player_idx_to_high_level_action[player_idx] = PUT_DOWN_DISH_ON_COUNTER
                    if player.get_object().name == 'soup':
                        player_idx_to_high_level_action[player_idx] = PUT_DOWN_SOUP_ON_COUNTER

                    new_state.add_object(player.remove_object(), i_pos)

                    try:
                        placed_item_uid = self.player_idx_to_item_holding_uid[player_idx]

                        self.item_tracking_dict[placed_item_uid]['player_holding'] = None
                        self.item_tracking_dict[placed_item_uid]['location_placed'] = i_pos

                        self.location_to_item_uid[i_pos] = placed_item_uid
                        self.item_uid_to_location[placed_item_uid] = i_pos
                        self.player_idx_to_item_holding_uid[player_idx] = None
                    except:
                        pass



                elif not player.has_object() and new_state.has_object(i_pos):
                    # Action Type 2: Player picked object up from counter
                    player.set_object(new_state.remove_object(i_pos))

                    if n_features == 7:
                        if tuple(i_pos) in shared_counters:
                            reward_featurized_state[6] = 0.15

                    if player.get_object().name == 'onion':
                        player_idx_to_high_level_action[player_idx] = PICKUP_ONION_FROM_COUNTER
                    if player.get_object().name == 'dish':
                        player_idx_to_high_level_action[player_idx] = PICKUP_DISH_FROM_COUNTER
                    if player.get_object().name == 'soup':
                        player_idx_to_high_level_action[player_idx] = PICKUP_SOUP_FROM_COUNTER

                    try:
                        picked_item_uid = self.location_to_item_uid[i_pos]

                        self.item_tracking_dict[picked_item_uid]['player_holding'] = player_idx
                        self.item_tracking_dict[picked_item_uid]['location_placed'] = None
                        self.item_tracking_dict[picked_item_uid]['past_players'].append(player_idx)
                        self.item_tracking_dict[picked_item_uid]['n_touches'] += 1

                        # if len(np.unique(self.item_tracking_dict[picked_item_uid]['past_players'])) > 1 \
                        #         and self.item_tracking_dict[picked_item_uid]['handoff_time'] <= self.mean_handoff_time:
                        #     shaped_reward += HANDOFF_TIME_REWARD
                        #     reward_featurized_state[5] = 1
                        # if len(np.unique(self.item_tracking_dict[picked_item_uid]['past_players'])) > 1 \
                        #         and self.item_tracking_dict[picked_item_uid]['handoff_time'] > self.mean_handoff_time:
                        #     if self.mean_handoff_time > 3:
                        #         shaped_reward -= HANDOFF_TIME_REWARD
                        # if len(np.unique(self.item_tracking_dict[picked_item_uid]['past_players'])) > 1 \
                        #         and self.item_tracking_dict[picked_item_uid]['handoff_time'] > FC_MEAN_TRANSFER:
                        #     shaped_reward += HANDOFF_TIME_OVER_MEAN_PENALTY

                        # if self.item_tracking_dict[picked_item_uid]['n_touches'] == len(np.unique(self.item_tracking_dict[picked_item_uid]['past_players'])):
                        #     # shaped_reward += NUM_TOUCHES_PENALTY
                        #     reward_featurized_state[6] = 1

                        # if self.item_tracking_dict[picked_item_uid]['n_touches'] > 4:
                        #     shaped_reward += OVERPASSING_PENALTY


                        self.location_to_item_uid[i_pos] = None
                        self.item_uid_to_location[picked_item_uid] = None
                        self.player_idx_to_item_holding_uid[player_idx] = picked_item_uid
                    except:
                        pass


            elif terrain_type == 'O' and player.held_object is None:
                # Action Type 3: Player picked up onion from dispenser

                player.set_object(ObjectState('onion', pos))

                player_idx_to_high_level_action[player_idx] = PICKUP_ONION_FROM_DISPENSER
                # **ADDED **** Player picked up an onion
                # If there were two full pots
                # if full_pots == num_pots:
                #     shaped_reward -= self.reward_shaping_params["DISH_PICKUP_REWARD"]

                # If player picked up an onion from dispenser
                try:
                    self.object_uid_counter += 1
                    new_obj_record = {
                        'player_holding': player_idx,
                        'past_players': [player_idx],
                        'location_placed': None,
                        'name': 'onion',
                        'limbo_time': 0,
                        'handoff_time': 0,
                        'completed': False,
                        'n_touches': 1
                    }
                    self.item_tracking_dict[self.object_uid_counter] = new_obj_record
                    self.player_idx_to_item_holding_uid[player_idx] = self.object_uid_counter
                    self.active_item_uids.append(self.object_uid_counter)
                except:
                    pass


            elif terrain_type == 'T' and player.held_object is None:
                # Action Type 4: Player picked up tomato from dispenser
                player.set_object(ObjectState('tomato', pos))

                player_idx_to_high_level_action[player_idx] = PICKUP_ONION_FROM_DISPENSER

                # If player picked up an tomato from dispenser
                self.object_uid_counter += 1
                new_obj_record = {
                    'player_holding': player_idx,
                    'past_players': [player_idx],
                    'location_placed': None,
                    'name': 'tomato',
                    'limbo_time': 0,
                    'handoff_time': 0,
                    'completed': False,
                    'n_touches': 1
                }
                self.item_tracking_dict[self.object_uid_counter] = new_obj_record
                self.player_idx_to_item_holding_uid[player_idx] = self.object_uid_counter
                self.active_item_uids.append(self.object_uid_counter)


            elif terrain_type == 'D' and player.held_object is None:
                # Action Type 5: Player picked up dish from dispenser
                dishes_already = len(new_state.player_objects_by_type['dish'])
                player.set_object(ObjectState('dish', pos))

                player_idx_to_high_level_action[player_idx] = PICKUP_DISH_FROM_DISPENSER

                dishes_on_counters = self.get_counter_objects_dict(new_state)["dish"]
                if len(nearly_ready_pots) > dishes_already and len(dishes_on_counters) == 0:
                    shaped_reward += self.reward_shaping_params["DISH_PICKUP_REWARD"]
                    reward_featurized_state[2] = 1
                # If player picked up an dish from dispenser
                self.object_uid_counter += 1
                new_obj_record = {
                    'player_holding': player_idx,
                    'past_players': [player_idx],
                    'location_placed': None,
                    'name': 'dish',
                    'limbo_time': 0,
                    'handoff_time': 0,
                    'completed': False,
                    'n_touches': 1
                }
                self.item_tracking_dict[self.object_uid_counter] = new_obj_record
                self.player_idx_to_item_holding_uid[player_idx] = self.object_uid_counter
                self.active_item_uids.append(self.object_uid_counter)

            elif terrain_type == 'P' and player.has_object():
                if player.get_object().name == 'dish' and new_state.has_object(i_pos):
                    # Action Type 6: Player picked up soup from pot with dish
                    player_idx_to_high_level_action[player_idx] = PICKUP_SOUP_W_DISH
                    obj = new_state.get_object(i_pos)
                    assert obj.name == 'soup', 'Object in pot was not soup'
                    _, num_items, cook_time = obj.state
                    if num_items == self.num_items_for_soup and cook_time >= self.soup_cooking_time:
                        player.remove_object()  # Turn the dish into the soup
                        player.set_object(new_state.remove_object(i_pos))
                        shaped_reward += self.reward_shaping_params["SOUP_PICKUP_FROM_READY_POT_REWARD"]
                        reward_featurized_state[3] = 1
                elif player.get_object().name in ['onion', 'tomato']:
                    item_type = player.get_object().name

                    if not new_state.has_object(i_pos):
                        # Action Type 7: Player placed onion or tomato in empty pot
                        player_idx_to_high_level_action[player_idx] = PUT_DOWN_ONION_IN_POT
                        # Pot was empty
                        player.remove_object()
                        new_state.add_object(ObjectState('soup', i_pos, (item_type, 1, 0)), i_pos)
                        shaped_reward += self.reward_shaping_params["ONION_IN_EMPTY_POT_REWARD"]
                        reward_featurized_state[0] = 1
                        # Onion placed in pot is no longer active
                        try:
                            placed_item_uid = self.player_idx_to_item_holding_uid[player_idx]

                            self.item_tracking_dict[placed_item_uid]['player_holding'] = None
                            self.item_tracking_dict[placed_item_uid]['location_placed'] = i_pos
                            self.item_tracking_dict[placed_item_uid]['completed'] = True

                            self.player_idx_to_item_holding_uid[player_idx] = None

                            self.active_item_uids.remove(placed_item_uid)
                        except:
                            pass


                    else:
                        # Action Type 8: Player placed onion in partially filled pot
                        # Pot has already items in it

                        obj = new_state.get_object(i_pos)
                        assert obj.name == 'soup', 'Object in pot was not soup'
                        soup_type, num_items, cook_time = obj.state
                        if num_items < self.num_items_for_soup and soup_type == item_type:
                            player.remove_object()
                            obj.state = (soup_type, num_items + 1, 0)
                            shaped_reward += self.reward_shaping_params["ONION_IN_PARTIAL_POT_REWARD"]
                            reward_featurized_state[1] = 1
                            current_pots_states_dict = self.get_pot_states(new_state)
                            num_cooking_pots = len(current_pots_states_dict['onion']['cooking'])

                            player_idx_to_high_level_action[player_idx] = PUT_DOWN_ONION_IN_POT

                            reward_featurized_state[1] = 1
                            if num_cooking_pots > 1:
                                shaped_reward += HIGH_POT_COVERAGE_SIMUL_COOKING_REWARD
                                reward_featurized_state[4] = 1

                            # Onion placed in pot is no longer active
                            try:
                                placed_item_uid = self.player_idx_to_item_holding_uid[player_idx]

                                self.item_tracking_dict[placed_item_uid]['player_holding'] = None
                                self.item_tracking_dict[placed_item_uid]['location_placed'] = i_pos
                                self.item_tracking_dict[placed_item_uid]['completed'] = True

                                self.player_idx_to_item_holding_uid[player_idx] = None

                                self.active_item_uids.remove(placed_item_uid)
                            except:
                                pass
                        # else:
                        #     player_idx_to_high_level_action[player_idx] = 1000

            elif terrain_type == 'S' and player.has_object():
                obj = player.get_object()
                if obj.name == 'soup':
                    # Action Type 9: Player delivered soup
                    player_idx_to_high_level_action[player_idx] = SERVE_SOUP
                    new_state, delivery_rew = self.deliver_soup(new_state, player, obj)
                    sparse_reward += delivery_rew
                    reward_featurized_state[5] = 1
                    # Dish served is no longer active
                    try:
                        placed_item_uid = self.player_idx_to_item_holding_uid[player_idx]

                        self.item_tracking_dict[placed_item_uid]['player_holding'] = None
                        self.item_tracking_dict[placed_item_uid]['location_placed'] = None
                        self.item_tracking_dict[placed_item_uid]['completed'] = True

                        self.player_idx_to_item_holding_uid[player_idx] = None

                        self.active_item_uids.remove(placed_item_uid)
                    except:
                        pass

                    # If last soup necessary was delivered, stop resolving interacts
                    if new_state.order_list is not None and len(new_state.order_list) == 0:
                        break

                # else:
                #     player_idx_to_high_level_action[player_idx] = 1000

            else:
                player_idx_to_high_level_action[player_idx] = STAY
        # Make state transition
        collided = self.resolve_movement(new_state, joint_action)
        # Finally, environment effects
        self.step_environment_effects(new_state)
        return player_idx_to_high_level_action, reward_featurized_state, sparse_reward

    def deliver_soup(self, state, player, soup_obj):
        """
        Deliver the soup, and get reward if there is no order list
        or if the type of the delivered soup matches the next order.
        """
        soup_type, num_items, cook_time = soup_obj.state
        assert soup_type in ObjectState.SOUP_TYPES
        assert num_items == self.num_items_for_soup
        assert cook_time >= self.soup_cooking_time, "Cook time {} mdp cook time {}".format(cook_time,
                                                                                           self.soup_cooking_time)
        player.remove_object()

        if state.order_list is None:
            return state, self.delivery_reward

        # If the delivered soup is the one currently required
        assert not self.is_terminal(state)
        current_order = state.order_list[0]
        if current_order == 'any' or soup_type == current_order:
            state.order_list = state.order_list[1:]
            return state, self.delivery_reward

        return state, 0

    def resolve_movement(self, state, joint_action):
        """Resolve player movement and deal with possible collisions"""
        new_positions, new_orientations, collided = self.compute_new_positions_and_orientations(state.players, joint_action)
        for player_state, new_pos, new_o in zip(state.players, new_positions, new_orientations):
            player_state.update_pos_and_or(new_pos, new_o)
        return collided

    def compute_new_positions_and_orientations(self, old_player_states, joint_action):
        """Compute new positions and orientations ignoring collisions"""
        new_positions, new_orientations = list(zip(*[
            self._move_if_direction(p.position, p.orientation, a) \
            for p, a in zip(old_player_states, joint_action)]))
        old_positions = tuple(p.position for p in old_player_states)
        new_positions, collided = self._handle_collisions(old_positions, new_positions)
        return new_positions, new_orientations, collided

    def is_transition_collision(self, old_positions, new_positions):
        # Checking for any players ending in same square
        if self.is_joint_position_collision(new_positions):
            return True
        # Check if any two players crossed paths
        for idx0, idx1 in itertools.combinations(range(self.num_players), 2):
            p1_old, p2_old = old_positions[idx0], old_positions[idx1]
            p1_new, p2_new = new_positions[idx0], new_positions[idx1]
            if p1_new == p2_old and p1_old == p2_new:
                return True
        return False

    def is_joint_position_collision(self, joint_position):
        return any(pos0 == pos1 for pos0, pos1 in itertools.combinations(joint_position, 2))

    def step_environment_effects(self, state):
        reward = 0
        for obj in state.objects.values():
            if obj.name == 'soup':
                x, y = obj.position
                soup_type, num_items, cook_time = obj.state
                # NOTE: cook_time is capped at self.soup_cooking_time
                if self.terrain_mtx[y][x] == 'P' and \
                        num_items == self.num_items_for_soup and \
                        cook_time < self.soup_cooking_time:
                    obj.state = soup_type, num_items, cook_time + 1
        return reward

    def _handle_collisions(self, old_positions, new_positions):
        """If agents collide, they stay at their old locations"""
        collided = False
        if self.is_transition_collision(old_positions, new_positions):
            collided = True
            return old_positions, collided
        return new_positions, collided

    def _get_terrain_type_pos_dict(self):
        pos_dict = defaultdict(list)
        for y, terrain_row in enumerate(self.terrain_mtx):
            for x, terrain_type in enumerate(terrain_row):
                pos_dict[terrain_type].append((x, y))
        return pos_dict

    def _move_if_direction(self, position, orientation, action):
        """Returns position and orientation that would
        be obtained after executing action"""
        if action == Action.INTERACT:
            return position, orientation
        new_pos = Action.move_in_direction(position, action)
        new_orientation = orientation if action == Action.STAY else action
        if new_pos not in self.get_valid_player_positions():
            return position, new_orientation
        return new_pos, new_orientation

    def _check_valid_state(self, state):
        """Checks that the state is valid.

        Conditions checked:
        - Players are on free spaces, not terrain
        - Held objects have the same position as the player holding them
        - Non-held objects are on terrain
        - No two players or non-held objects occupy the same position
        - Objects have a valid state (eg. no pot with 4 onions)
        """
        all_objects = list(state.objects.values())
        for player_state in state.players:
            # Check that players are not on terrain
            pos = player_state.position
            assert pos in self.get_valid_player_positions()

            # Check that held objects have the same position
            if player_state.held_object is not None:
                all_objects.append(player_state.held_object)
                assert player_state.held_object.position == player_state.position

        for obj_pos, obj_state in state.objects.items():
            # Check that the hash key position agrees with the position stored
            # in the object state
            assert obj_state.position == obj_pos
            # Check that non-held objects are on terrain
            assert self.get_terrain_type_at_pos(obj_pos) != ' '

        # Check that players and non-held objects don't overlap
        all_pos = [player_state.position for player_state in state.players]
        all_pos += [obj_state.position for obj_state in state.objects.values()]
        assert len(all_pos) == len(set(all_pos)), "Overlapping players or objects"

        # Check that objects have a valid state
        for obj_state in all_objects:
            assert obj_state.is_valid()

    @staticmethod
    def _assert_valid_grid(grid):
        """Raises an AssertionError if the grid is invalid.

        grid:  A sequence of sequences of spaces, representing a grid of a
        certain height and width. grid[y][x] is the space at row y and column
        x. A space must be either 'X' (representing a counter), ' ' (an empty
        space), 'O' (onion supply), 'P' (pot), 'D' (dish supply), 'S' (serving
        location), '1' (player 1) and '2' (player 2).
        """
        height = len(grid)
        width = len(grid[0])

        # Make sure the grid is not ragged
        assert all(len(row) == width for row in grid), 'Ragged grid'

        # Borders must not be free spaces
        def is_not_free(c):
            return c in 'XOPDST'

        for y in range(height):
            assert is_not_free(grid[y][0]), 'Left border must not be free'
            assert is_not_free(grid[y][-1]), 'Right border must not be free'
        for x in range(width):
            assert is_not_free(grid[0][x]), 'Top border must not be free'
            assert is_not_free(grid[-1][x]), 'Bottom border must not be free'

        all_elements = [element for row in grid for element in row]
        digits = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        layout_digits = [e for e in all_elements if e in digits]
        num_players = len(layout_digits)
        assert num_players > 0, "No players (digits) in grid"
        layout_digits = list(sorted(map(int, layout_digits)))
        assert layout_digits == list(range(1, num_players + 1)), "Some players were missing"

        assert all(c in 'XOPDST123456789 ' for c in all_elements), 'Invalid character in grid'
        assert all_elements.count('1') == 1, "'1' must be present exactly once"
        assert all_elements.count('D') >= 1, "'D' must be present at least once"
        assert all_elements.count('S') >= 1, "'S' must be present at least once"
        assert all_elements.count('P') >= 1, "'P' must be present at least once"
        assert all_elements.count('O') >= 1 or all_elements.count('T') >= 1, "'O' or 'T' must be present at least once"

    #####################
    # TERMINAL GRAPHICS #
    #####################

    def state_string(self, state):
        """String representation of the current state"""
        players_dict = {player.position: player for player in state.players}

        grid_string = ""
        for y, terrain_row in enumerate(self.terrain_mtx):
            for x, element in enumerate(terrain_row):
                if (x, y) in players_dict.keys():
                    player = players_dict[(x, y)]
                    orientation = player.orientation
                    assert orientation in Direction.ALL_DIRECTIONS

                    grid_string += Action.ACTION_TO_CHAR[orientation]
                    player_object = player.held_object
                    if player_object:
                        grid_string += player_object.name[:1]
                    else:
                        player_idx_lst = [i for i, p in enumerate(state.players) if p.position == player.position]
                        assert len(player_idx_lst) == 1
                        grid_string += str(player_idx_lst[0])
                else:
                    if element == "X" and state.has_object((x, y)):
                        state_obj = state.get_object((x, y))
                        grid_string = grid_string + element + state_obj.name[:1]

                    elif element == "P" and state.has_object((x, y)):
                        soup_obj = state.get_object((x, y))
                        soup_type, num_items, cook_time = soup_obj.state
                        if soup_type == "onion":
                            grid_string += "ø"
                        elif soup_type == "tomato":
                            grid_string += "†"
                        else:
                            raise ValueError()

                        if num_items == self.num_items_for_soup:
                            grid_string += str(cook_time)

                        # NOTE: do not currently have terminal graphics
                        # support for cooking times greater than 3.
                        elif num_items == 2:
                            grid_string += "="
                        else:
                            grid_string += "-"
                    else:
                        grid_string += element + " "

            grid_string += "\n"

        if state.order_list is not None:
            grid_string += "Current orders: {}/{} are any's\n".format(
                len(state.order_list), len([order == "any" for order in state.order_list])
            )
        return grid_string

    ###################
    # STATE ENCODINGS #
    ###################

    def lossless_state_encoding(self, overcooked_state, debug=False):
        """Featurizes a OvercookedState object into a stack of boolean masks that are easily readable by a CNN"""
        assert type(debug) is bool
        base_map_features = ["pot_loc", "counter_loc", "onion_disp_loc", "dish_disp_loc", "serve_loc"]
        variable_map_features = ["onions_in_pot", "onions_cook_time", "onion_soup_loc", "dishes", "onions"]

        all_objects = overcooked_state.all_objects_list

        def make_layer(position, value):
            layer = np.zeros(self.shape)
            layer[position] = value
            return layer

        def process_for_player(primary_agent_idx):
            # Ensure that primary_agent_idx layers are ordered before other_agent_idx layers
            other_agent_idx = 1 - primary_agent_idx
            ordered_player_features = ["player_{}_loc".format(primary_agent_idx),
                                       "player_{}_loc".format(other_agent_idx)] + \
                                      ["player_{}_orientation_{}".format(i, Direction.DIRECTION_TO_INDEX[d])
                                       for i, d in itertools.product([primary_agent_idx, other_agent_idx],
                                                                     Direction.ALL_DIRECTIONS)]

            LAYERS = ordered_player_features + base_map_features + variable_map_features
            state_mask_dict = {k: np.zeros(self.shape) for k in LAYERS}

            # MAP LAYERS
            for loc in self.get_counter_locations():
                state_mask_dict["counter_loc"][loc] = 1

            for loc in self.get_pot_locations():
                state_mask_dict["pot_loc"][loc] = 1

            for loc in self.get_onion_dispenser_locations():
                state_mask_dict["onion_disp_loc"][loc] = 1

            for loc in self.get_dish_dispenser_locations():
                state_mask_dict["dish_disp_loc"][loc] = 1

            for loc in self.get_serving_locations():
                state_mask_dict["serve_loc"][loc] = 1

            # PLAYER LAYERS
            for i, player in enumerate(overcooked_state.players):
                player_orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
                state_mask_dict["player_{}_loc".format(i)] = make_layer(player.position, 1)
                state_mask_dict["player_{}_orientation_{}".format(i, player_orientation_idx)] = make_layer(
                    player.position, 1)

            # OBJECT & STATE LAYERS
            for obj in all_objects:
                if obj.name == "soup":
                    soup_type, num_onions, cook_time = obj.state
                    if soup_type == "onion":
                        if obj.position in self.get_pot_locations():
                            soup_type, num_onions, cook_time = obj.state
                            state_mask_dict["onions_in_pot"] += make_layer(obj.position, num_onions)
                            state_mask_dict["onions_cook_time"] += make_layer(obj.position, cook_time)
                        else:
                            # If player soup is not in a pot, put it in separate mask
                            state_mask_dict["onion_soup_loc"] += make_layer(obj.position, 1)
                    else:
                        raise ValueError("Unrecognized soup")

                elif obj.name == "dish":
                    state_mask_dict["dishes"] += make_layer(obj.position, 1)
                elif obj.name == "onion":
                    state_mask_dict["onions"] += make_layer(obj.position, 1)
                else:
                    raise ValueError("Unrecognized object")

            if debug:
                print(len(LAYERS))
                print(len(state_mask_dict))
                for k, v in state_mask_dict.items():
                    print(k)
                    print(np.transpose(v, (1, 0)))

            # Stack of all the state masks, order decided by order of LAYERS
            state_mask_stack = np.array([state_mask_dict[layer_id] for layer_id in LAYERS])
            state_mask_stack = np.transpose(state_mask_stack, (1, 2, 0))
            assert state_mask_stack.shape[:2] == self.shape
            assert state_mask_stack.shape[2] == len(LAYERS)
            # NOTE: currently not including time left or order_list in featurization
            return np.array(state_mask_stack).astype(int)

        # NOTE: Currently not very efficient, a decent amount of computation repeated here
        num_players = len(overcooked_state.players)
        final_obs_for_players = tuple(process_for_player(i) for i in range(num_players))
        return final_obs_for_players

    def featurize_state(self, overcooked_state, mlp):
        """
        Encode state with some manually designed features.
        NOTE: currently works for just two players.
        """

        all_features = {}

        def make_closest_feature(idx, name, locations):
            "Compute (x, y) deltas to closest feature of type `name`, and save it in the features dict"
            all_features["p{}_closest_{}".format(idx, name)] = self.get_deltas_to_closest_location(player, locations,
                                                                                                   mlp)

        IDX_TO_OBJ = ["onion", "soup", "dish"]
        OBJ_TO_IDX = {o_name: idx for idx, o_name in enumerate(IDX_TO_OBJ)}

        counter_objects = self.get_counter_objects_dict(overcooked_state)
        pot_state = self.get_pot_states(overcooked_state)

        # Player Info
        for i, player in enumerate(overcooked_state.players):
            orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
            all_features["p{}_orientation".format(i)] = np.eye(4)[orientation_idx]
            obj = player.held_object

            if obj is None:
                held_obj_name = "none"
                all_features["p{}_objs".format(i)] = np.zeros(len(IDX_TO_OBJ))
            else:
                held_obj_name = obj.name
                obj_idx = OBJ_TO_IDX[held_obj_name]
                all_features["p{}_objs".format(i)] = np.eye(len(IDX_TO_OBJ))[obj_idx]

            # Closest feature of each type
            if held_obj_name == "onion":
                all_features["p{}_closest_onion".format(i)] = (0, 0)
            else:
                make_closest_feature(i, "onion", self.get_onion_dispenser_locations() + counter_objects["onion"])

            make_closest_feature(i, "empty_pot", pot_state["empty"])
            make_closest_feature(i, "one_onion_pot", pot_state["onion"]["one_onion"])
            make_closest_feature(i, "two_onion_pot", pot_state["onion"]["two_onion"])
            make_closest_feature(i, "cooking_pot", pot_state["onion"]["cooking"])
            make_closest_feature(i, "ready_pot", pot_state["onion"]["ready"])

            if held_obj_name == "dish":
                all_features["p{}_closest_dish".format(i)] = (0, 0)
            else:
                make_closest_feature(i, "dish", self.get_dish_dispenser_locations() + counter_objects["dish"])

            if held_obj_name == "soup":
                all_features["p{}_closest_soup".format(i)] = (0, 0)
            else:
                make_closest_feature(i, "soup", counter_objects["soup"])

            make_closest_feature(i, "serving", self.get_serving_locations())

            for direction, pos_and_feat in enumerate(self.get_adjacent_features(player)):
                adj_pos, feat = pos_and_feat

                if direction == player.orientation:
                    # Check if counter we are facing is empty
                    facing_counter = (feat == 'X' and adj_pos not in overcooked_state.objects.keys())
                    facing_counter_feature = [1] if facing_counter else [0]
                    all_features["p{}_facing_empty_counter".format(i)] = facing_counter_feature

                all_features["p{}_wall_{}".format(i, direction)] = [0] if feat == ' ' else [1]

        features_np = {k: np.array(v) for k, v in all_features.items()}

        p0, p1 = overcooked_state.players
        p0_dict = {k: v for k, v in features_np.items() if k[:2] == "p0"}
        p1_dict = {k: v for k, v in features_np.items() if k[:2] == "p1"}
        p0_features = np.concatenate(list(p0_dict.values()))
        p1_features = np.concatenate(list(p1_dict.values()))

        p1_rel_to_p0 = np.array(pos_distance(p1.position, p0.position))
        abs_pos_p0 = np.array(p0.position)
        ordered_features_p0 = np.squeeze(np.concatenate([p0_features, p1_features, p1_rel_to_p0, abs_pos_p0]))

        p0_rel_to_p1 = np.array(pos_distance(p0.position, p1.position))
        abs_pos_p1 = np.array(p0.position)
        ordered_features_p1 = np.squeeze(np.concatenate([p1_features, p0_features, p0_rel_to_p1, abs_pos_p1]))
        return ordered_features_p0, ordered_features_p1


    def featurize_state_new(self, overcooked_state, mlp, prev_joint_action):
        """
        Encode state with some manually designed features.
        NOTE: currently works for just two players.
        """

        all_features = {}

        def make_closest_feature(idx, name, locations):
            "Compute (x, y) deltas to closest feature of type `name`, and save it in the features dict"
            all_features["p{}_closest_{}".format(idx, name)] = self.get_deltas_to_closest_location(player, locations,
                                                                                                   mlp)

        IDX_TO_OBJ = ["onion", "soup", "dish"]
        OBJ_TO_IDX = {o_name: idx for idx, o_name in enumerate(IDX_TO_OBJ)}

        counter_objects = self.get_counter_objects_dict(overcooked_state)
        pot_state = self.get_pot_states(overcooked_state)

        # pot_tracker = {}
        # for location in self.get_pot_locations():
        #     pot_tracker[location] = {}
        #     pot_tracker[location]['contents'] = []
        #     # pot_tracker[location]['state'] = EMPTY
        #
        # num_items_per_pot = []
        # num_ready_pots = 0
        # for pot_pos in self.get_pot_locations():
        #     if not overcooked_state.has_object(pot_pos):
        #         pot_tracker[pot_pos]['contents'] = []
        #         # pot_tracker[pot_pos]['state'] = EMPTY
        #     else:
        #         soup_obj = overcooked_state.get_object(pot_pos)
        #         soup_type, num_items, cook_time = soup_obj.state
        #         if 0 < num_items < self.num_items_for_soup:
        #             pot_tracker[pot_pos]['contents'] = ['onion' for _ in range(num_items)]
        #             # pot_tracker[pot_pos]['state'] = PARTIALLY_FILLED
        #         elif num_items == self.num_items_for_soup:
        #             if cook_time >= self.soup_cooking_time:
        #                 pot_tracker[pot_pos]['contents'] = ['onion' for _ in range(num_items)]
        #                 # pot_tracker[pot_pos]['state'] = READY
        #                 num_ready_pots += 1
        #             else:
        #                 pot_tracker[pot_pos]['contents'] = ['onion' for _ in range(num_items)]
        #                 # pot_tracker[pot_pos]['state'] = COOKING
        #     num_items_per_pot.append(len(pot_tracker[pot_pos]['contents']))
        #
        # counters_considered = self.terrain_pos_dict['X']
        # free_onion_tracker = []
        # free_dish_tracker = []
        # free_soup_tracker = []
        # valid_passing_counters = [(2, 1), (2, 2), (2, 3)]
        # for obj in overcooked_state.objects.values():
        #     if obj.position in counters_considered and obj.position in valid_passing_counters:
        #         # counter_objects_dict[obj.name].append(obj.position)
        #         if obj.name == 'onion':
        #             free_onion_tracker.append(obj.position)
        #         if obj.name == 'dish':
        #             free_dish_tracker.append(obj.position)
        #         if obj.name == 'soup':
        #             free_soup_tracker.append(obj.position)

        # Player Info
        for i, player in enumerate(overcooked_state.players):
            # add info about passing counters
            # all_features["p{}_free_onion".format(i)] = [1] if len(free_onion_tracker) > 0 else [0]
            # all_features["p{}_free_dish".format(i)] = [1] if len(free_dish_tracker) > 0 else [0]
            # all_features["p{}_free_soup".format(i)] = [1] if len(free_soup_tracker) > 0 else [0]
            # all_features["p{}_num_items_per_pot".format(i)] = tuple(num_items_per_pot)
            # all_features["p{}_num_ready_pots".format(i)] = [num_ready_pots]
            #
            #
            opp_idx = 1-i
            orientation_idx = Direction.DIRECTION_TO_INDEX[overcooked_state.players[opp_idx].orientation]
            all_features["p{}_opp_orientation".format(i)] = np.eye(4)[orientation_idx]
            all_features["p{}_opp_position".format(i)] = overcooked_state.players[opp_idx].position

            partner_action = prev_joint_action[opp_idx]
            # print("partner_action", partner_action)
            partner_action_idx = Action.ACTION_TO_INDEX[partner_action]
            partner_action_idx = 0
            all_features["p{}_opp_prev_action".format(i)] = np.eye(6)[partner_action_idx]


            orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
            all_features["p{}_orientation".format(i)] = np.eye(4)[orientation_idx]
            all_features["p{}_position".format(i)] = player.position

            obj = player.held_object

            if obj is None:
                held_obj_name = "none"
                all_features["p{}_objs".format(i)] = np.zeros(len(IDX_TO_OBJ))
            else:
                held_obj_name = obj.name
                obj_idx = OBJ_TO_IDX[held_obj_name]
                all_features["p{}_objs".format(i)] = np.eye(len(IDX_TO_OBJ))[obj_idx]

            # Closest feature of each type
            if held_obj_name == "onion":
                all_features["p{}_closest_onion".format(i)] = (0, 0)
            else:
                make_closest_feature(i, "onion", self.get_onion_dispenser_locations() + counter_objects["onion"])

            make_closest_feature(i, "empty_pot", pot_state["empty"])
            make_closest_feature(i, "one_onion_pot", pot_state["onion"]["one_onion"])
            make_closest_feature(i, "two_onion_pot", pot_state["onion"]["two_onion"])
            make_closest_feature(i, "cooking_pot", pot_state["onion"]["cooking"])
            make_closest_feature(i, "ready_pot", pot_state["onion"]["ready"])

            if held_obj_name == "dish":
                all_features["p{}_closest_dish".format(i)] = (0, 0)
            else:
                make_closest_feature(i, "dish", self.get_dish_dispenser_locations() + counter_objects["dish"])



            if held_obj_name == "soup":
                all_features["p{}_closest_soup".format(i)] = (0, 0)
            else:
                make_closest_feature(i, "soup", counter_objects["soup"])

            make_closest_feature(i, "serving", self.get_serving_locations())

            for direction, pos_and_feat in enumerate(self.get_adjacent_features(player)):
                adj_pos, feat = pos_and_feat
                # print("direction", (direction, player.orientation))
                if direction == Direction.DIRECTION_TO_INDEX[player.orientation]:
                    # print("direction", (direction, player.orientation))
                    # Check if counter we are facing is empty
                    facing_counter = (feat == 'X' and adj_pos not in overcooked_state.objects.keys())
                    facing_counter_feature = [1] if facing_counter else [0]
                    all_features["p{}_facing_empty_counter".format(i)] = facing_counter_feature

                all_features["p{}_wall_{}".format(i, direction)] = [0] if feat == ' ' else [1]

        features_np = {k: np.array(v) for k, v in all_features.items()}

        p0, p1 = overcooked_state.players
        p0_dict = {k: v for k, v in features_np.items() if k[:2] == "p0"}
        p1_dict = {k: v for k, v in features_np.items() if k[:2] == "p1"}
        p0_features = np.concatenate(list(p0_dict.values()))
        p1_features = np.concatenate(list(p1_dict.values()))

        p1_rel_to_p0 = np.array(pos_distance(p1.position, p0.position))
        abs_pos_p0 = np.array(p0.position)
        ordered_features_p0 = np.squeeze(np.concatenate([p0_features, p1_features, p1_rel_to_p0, abs_pos_p0]))

        p0_rel_to_p1 = np.array(pos_distance(p0.position, p1.position))
        abs_pos_p1 = np.array(p0.position)
        ordered_features_p1 = np.squeeze(np.concatenate([p1_features, p0_features, p0_rel_to_p1, abs_pos_p1]))
        return ordered_features_p0, ordered_features_p1

    def featurize_state_complex(self, overcooked_state, mlp, prev_joint_action):
        """
        Encode state with some manually designed features.
        NOTE: currently works for just two players.
        """

        all_features = {}

        def make_closest_feature(idx, name, locations):
            "Compute (x, y) deltas to closest feature of type `name`, and save it in the features dict"
            all_features["p{}_closest_{}".format(idx, name)] = self.get_deltas_to_closest_location(player, locations,
                                                                                                   mlp)

        IDX_TO_OBJ = ["onion", "soup", "dish"]
        OBJ_TO_IDX = {o_name: idx for idx, o_name in enumerate(IDX_TO_OBJ)}

        counter_objects = self.get_counter_objects_dict(overcooked_state)
        pot_state = self.get_pot_states(overcooked_state)

        pot_tracker = {}
        for location in self.get_pot_locations():
            pot_tracker[location] = {}
            pot_tracker[location]['contents'] = []
            # pot_tracker[location]['state'] = EMPTY

        num_items_per_pot = []
        num_ready_pots = 0
        for pot_pos in self.get_pot_locations():
            if not overcooked_state.has_object(pot_pos):
                pot_tracker[pot_pos]['contents'] = []
                # pot_tracker[pot_pos]['state'] = EMPTY
            else:
                soup_obj = overcooked_state.get_object(pot_pos)
                soup_type, num_items, cook_time = soup_obj.state
                if 0 < num_items < self.num_items_for_soup:
                    pot_tracker[pot_pos]['contents'] = ['onion' for _ in range(num_items)]
                    # pot_tracker[pot_pos]['state'] = PARTIALLY_FILLED
                elif num_items == self.num_items_for_soup:
                    if cook_time >= self.soup_cooking_time:
                        pot_tracker[pot_pos]['contents'] = ['onion' for _ in range(num_items)]
                        # pot_tracker[pot_pos]['state'] = READY
                        num_ready_pots += 1
                    else:
                        pot_tracker[pot_pos]['contents'] = ['onion' for _ in range(num_items)]
                        # pot_tracker[pot_pos]['state'] = COOKING
            num_items_per_pot.append(len(pot_tracker[pot_pos]['contents']))

        counters_considered = self.terrain_pos_dict['X']
        free_onion_tracker_valid_pass = []
        free_dish_tracker_valid_pass = []
        free_soup_tracker_valid_pass = []

        free_onion_tracker_p0_counter = []
        free_dish_tracker_p0_counter = []
        free_soup_tracker_p0_counter = []

        free_onion_tracker_p1_counter = []
        free_dish_tracker_p1_counter = []
        free_soup_tracker_p1_counter = []

        valid_passing_counters = [(2, 1), (2, 2), (2, 3)]
        p0_personal_counters = [(4, 2), (4, 4)]
        p1_personal_counters = [(1, 0), (1, 4)]

        for obj in overcooked_state.objects.values():
            if obj.position in counters_considered:
                # counter_objects_dict[obj.name].append(obj.position)
                if obj.position in valid_passing_counters:
                    if obj.name == 'onion':
                        free_onion_tracker_valid_pass.append(obj.position)
                    if obj.name == 'dish':
                        free_dish_tracker_valid_pass.append(obj.position)
                    if obj.name == 'soup':
                        free_soup_tracker_valid_pass.append(obj.position)

                elif obj.position in p0_personal_counters:
                    if obj.name == 'onion':
                        free_onion_tracker_p0_counter.append(obj.position)
                    if obj.name == 'dish':
                        free_dish_tracker_p0_counter.append(obj.position)
                    if obj.name == 'soup':
                        free_soup_tracker_p0_counter.append(obj.position)

                elif obj.position in p1_personal_counters:
                    if obj.name == 'onion':
                        free_onion_tracker_p1_counter.append(obj.position)
                    if obj.name == 'dish':
                        free_dish_tracker_p1_counter.append(obj.position)
                    if obj.name == 'soup':
                        free_soup_tracker_p1_counter.append(obj.position)

        # Player Info
        for i, player in enumerate(overcooked_state.players):
            # add info about passing counters
            all_features["p{}_free_onion_valid_pass".format(i)] = [len(free_onion_tracker_valid_pass)]
            all_features["p{}_free_dish_valid_pass".format(i)] = [len(free_dish_tracker_valid_pass)]
            all_features["p{}_free_soup_valid_pass".format(i)] = [len(free_dish_tracker_valid_pass)]

            all_features["p{}_free_onion_p0_counter".format(i)] = [len(free_onion_tracker_p0_counter)]
            all_features["p{}_free_dish_p0_counter".format(i)] = [len(free_dish_tracker_p0_counter)]
            all_features["p{}_free_soup_p0_counter".format(i)] = [len(free_dish_tracker_p0_counter)]

            all_features["p{}_free_onion_p1_counter".format(i)] = [len(free_onion_tracker_p1_counter)]
            all_features["p{}_free_dish_p1_counter".format(i)] = [len(free_dish_tracker_p1_counter)]
            all_features["p{}_free_soup_p1_counter".format(i)] = [len(free_dish_tracker_p1_counter)]



            all_features["p{}_num_items_per_pot".format(i)] = tuple(num_items_per_pot)
            all_features["p{}_num_ready_pots".format(i)] = [num_ready_pots]
            #
            #
            opp_idx = 1-i
            orientation_idx = Direction.DIRECTION_TO_INDEX[overcooked_state.players[opp_idx].orientation]
            all_features["p{}_opp_orientation".format(i)] = np.eye(4)[orientation_idx]
            all_features["p{}_opp_position".format(i)] = overcooked_state.players[opp_idx].position

            partner_action = prev_joint_action[opp_idx]
            # print("partner_action", partner_action)
            partner_action_idx = Action.ACTION_TO_INDEX[partner_action]
            partner_action_idx = 0
            all_features["p{}_opp_prev_action".format(i)] = np.eye(6)[partner_action_idx]


            orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
            all_features["p{}_orientation".format(i)] = np.eye(4)[orientation_idx]
            all_features["p{}_position".format(i)] = player.position

            obj = player.held_object

            if obj is None:
                held_obj_name = "none"
                all_features["p{}_objs".format(i)] = np.zeros(len(IDX_TO_OBJ))
            else:
                held_obj_name = obj.name
                obj_idx = OBJ_TO_IDX[held_obj_name]
                all_features["p{}_objs".format(i)] = np.eye(len(IDX_TO_OBJ))[obj_idx]

            # Closest feature of each type
            if held_obj_name == "onion":
                all_features["p{}_closest_onion".format(i)] = (0, 0)
            else:
                make_closest_feature(i, "onion", self.get_onion_dispenser_locations() + counter_objects["onion"])

            make_closest_feature(i, "empty_pot", pot_state["empty"])
            make_closest_feature(i, "one_onion_pot", pot_state["onion"]["one_onion"])
            make_closest_feature(i, "two_onion_pot", pot_state["onion"]["two_onion"])
            make_closest_feature(i, "cooking_pot", pot_state["onion"]["cooking"])
            make_closest_feature(i, "ready_pot", pot_state["onion"]["ready"])

            if held_obj_name == "dish":
                all_features["p{}_closest_dish".format(i)] = (0, 0)
            else:
                make_closest_feature(i, "dish", self.get_dish_dispenser_locations() + counter_objects["dish"])



            if held_obj_name == "soup":
                all_features["p{}_closest_soup".format(i)] = (0, 0)
            else:
                make_closest_feature(i, "soup", counter_objects["soup"])

            make_closest_feature(i, "serving", self.get_serving_locations())

            facing_direction_counter_empty = None
            for direction, pos_and_feat in enumerate(self.get_adjacent_features(player)):
                adj_pos, feat = pos_and_feat
                # print("direction", (direction, player.orientation))
                if direction == Direction.DIRECTION_TO_INDEX[player.orientation]:
                    # print("direction", (direction, player.orientation))
                    # Check if counter we are facing is empty
                    facing_counter = (feat == 'X' and adj_pos not in overcooked_state.objects.keys())
                    facing_counter_feature = [1] if facing_counter else [0]
                    facing_direction_counter_empty = facing_counter_feature


                all_features["p{}_wall_{}".format(i, direction)] = [0] if feat == ' ' else [1]

            all_features["p{}_facing_empty_counter".format(i)] = facing_direction_counter_empty

        features_np = {k: np.array(v) for k, v in all_features.items()}

        p0, p1 = overcooked_state.players
        p0_dict = {k: v for k, v in features_np.items() if k[:2] == "p0"}
        p1_dict = {k: v for k, v in features_np.items() if k[:2] == "p1"}
        p0_features = np.concatenate(list(p0_dict.values()))
        p1_features = np.concatenate(list(p1_dict.values()))

        p1_rel_to_p0 = np.array(pos_distance(p1.position, p0.position))
        abs_pos_p0 = np.array(p0.position)
        ordered_features_p0 = np.squeeze(np.concatenate([p0_features, p1_features, p1_rel_to_p0, abs_pos_p0]))

        p0_rel_to_p1 = np.array(pos_distance(p0.position, p1.position))
        abs_pos_p1 = np.array(p0.position)
        ordered_features_p1 = np.squeeze(np.concatenate([p1_features, p0_features, p0_rel_to_p1, abs_pos_p1]))
        return ordered_features_p0, ordered_features_p1

    def featurize_state_for_irl(self, overcooked_state, mlp, prev_joint_action):
        """
        Encode state with some manually designed features.
        NOTE: currently works for just two players.
        State features


        1. P1 position
        2. P1 orientation
        3. P2 position
        4. P2 orientation
        5. P1 carrying
        6. P2 carrying
        7. Counter 1 state - none, onion, dish, soup
        8. Counter 2 state - none, onion, dish, soup
        9. Counter 3 state - none, onion, dish, soup
        10. Counter 4 state - none, onion, dish, soup
        11. Counter 5 state - none, onion, dish, soup
        12. Counter 6 state - none, onion, dish, soup
        13. Counter 7 state - none, onion, dish, soup
        14. Top Pot 1 state - empty, 1 onion, 2 onion, 3 onion cooking, 3 onion ready
        15. Right Pot 2 state - empty, 1 onion, 2 onion, 3 onion cooking, 3 onion ready
        """

        all_features = {}

        def make_closest_feature(idx, name, locations):
            "Compute (x, y) deltas to closest feature of type `name`, and save it in the features dict"
            all_features["p{}_closest_{}".format(idx, name)] = self.get_deltas_to_closest_location(player, locations,
                                                                                                   mlp)

        IDX_TO_OBJ = ["onion", "soup", "dish"]
        OBJ_TO_IDX = {o_name: idx for idx, o_name in enumerate(IDX_TO_OBJ)}

        # Pot states
        EMPTY = 0
        ONE_ONION = 1
        TWO_ONION = 2
        THREE_ONION_COOKING = 3
        THREE_ONION_READY = 4
        N_POT_STATES = 5

        counter_objects = self.get_counter_objects_dict(overcooked_state)
        pot_state = self.get_pot_states(overcooked_state)
        pot_locations = list(self.get_pot_locations())

        pot_tracker = {}
        for location in self.get_pot_locations():
            pot_tracker[location] = {}
            pot_tracker[location]['contents'] = []
            pot_tracker[location]['state'] = EMPTY

        num_items_per_pot = []
        num_ready_pots = 0
        for pot_pos in self.get_pot_locations():
            if not overcooked_state.has_object(pot_pos):
                pot_tracker[pot_pos]['contents'] = []
                pot_tracker[pot_pos]['state'] = EMPTY
            else:
                soup_obj = overcooked_state.get_object(pot_pos)
                soup_type, num_items, cook_time = soup_obj.state
                if 0 < num_items < self.num_items_for_soup:
                    pot_tracker[pot_pos]['contents'] = ['onion' for _ in range(num_items)]

                    if len(pot_tracker[pot_pos]['contents']) == 1:
                        pot_tracker[pot_pos]['state'] = ONE_ONION

                    if len(pot_tracker[pot_pos]['contents']) == 2:
                        pot_tracker[pot_pos]['state'] = TWO_ONION

                elif num_items == self.num_items_for_soup:
                    if cook_time >= self.soup_cooking_time:
                        pot_tracker[pot_pos]['contents'] = ['onion' for _ in range(num_items)]
                        pot_tracker[pot_pos]['state'] = THREE_ONION_READY
                        num_ready_pots += 1
                    else:
                        pot_tracker[pot_pos]['contents'] = ['onion' for _ in range(num_items)]
                        pot_tracker[pot_pos]['state'] = THREE_ONION_COOKING
            num_items_per_pot.append(len(pot_tracker[pot_pos]['contents']))

        counters_considered = self.terrain_pos_dict['X']

        valid_passing_counters = [(2, 1), (2, 2), (2, 3)]
        p0_personal_counters = [(4, 2), (4, 4)]
        p1_personal_counters = [(1, 0), (1, 4)]

        all_valid_counter_locations = [(2, 1), (2, 2), (2, 3), (4, 2), (4, 4), (1, 0), (1, 4)]

        # Counter states
        EMPTY = 0
        ONION = 1
        DISH = 2
        SOUP = 3
        N_COUNTER_STATES = 4



        counter_location_to_state = {}
        for counter_loc in all_valid_counter_locations:
            counter_location_to_state[counter_loc] = EMPTY

        for obj in overcooked_state.objects.values():
            if obj.position in counters_considered:
                # counter_objects_dict[obj.name].append(obj.position)

                if obj.name == 'onion':
                    counter_location_to_state[obj.position] = ONION
                if obj.name == 'dish':
                    counter_location_to_state[obj.position] = DISH
                if obj.name == 'soup':
                    counter_location_to_state[obj.position] = SOUP



        # Team state features Info
        for i, player in enumerate(overcooked_state.players):
            # Add player position and orientations
            orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
            all_features["p{}_orientation".format(i)] = np.eye(4)[orientation_idx]
            all_features["p{}_position".format(i)] = player.position

            obj = player.held_object

            if obj is None:
                held_obj_name = "none"
                all_features["p{}_objs".format(i)] = np.zeros(len(IDX_TO_OBJ))
            else:
                held_obj_name = obj.name
                obj_idx = OBJ_TO_IDX[held_obj_name]
                all_features["p{}_objs".format(i)] = np.eye(len(IDX_TO_OBJ))[obj_idx]

            # Closest feature of each type
            if held_obj_name == "onion":
                all_features["p{}_closest_onion".format(i)] = (0, 0)
            else:
                make_closest_feature(i, "onion", self.get_onion_dispenser_locations() + counter_objects["onion"])

            make_closest_feature(i, "empty_pot", pot_state["empty"])
            make_closest_feature(i, "one_onion_pot", pot_state["onion"]["one_onion"])
            make_closest_feature(i, "two_onion_pot", pot_state["onion"]["two_onion"])
            make_closest_feature(i, "cooking_pot", pot_state["onion"]["cooking"])
            make_closest_feature(i, "ready_pot", pot_state["onion"]["ready"])

            if held_obj_name == "dish":
                all_features["p{}_closest_dish".format(i)] = (0, 0)
            else:
                make_closest_feature(i, "dish", self.get_dish_dispenser_locations() + counter_objects["dish"])



            if held_obj_name == "soup":
                all_features["p{}_closest_soup".format(i)] = (0, 0)
            else:
                make_closest_feature(i, "soup", counter_objects["soup"])

            make_closest_feature(i, "serving", self.get_serving_locations())

            facing_direction_counter_empty = None
            for direction, pos_and_feat in enumerate(self.get_adjacent_features(player)):
                adj_pos, feat = pos_and_feat
                # print("direction", (direction, player.orientation))
                if direction == Direction.DIRECTION_TO_INDEX[player.orientation]:
                    # print("direction", (direction, player.orientation))
                    # Check if counter we are facing is empty
                    facing_counter = (feat == 'X' and adj_pos not in overcooked_state.objects.keys())
                    facing_counter_feature = [1] if facing_counter else [0]
                    facing_direction_counter_empty = facing_counter_feature


                all_features["p{}_wall_{}".format(i, direction)] = [0] if feat == ' ' else [1]

            all_features["p{}_facing_empty_counter".format(i)] = facing_direction_counter_empty

        for i, c_loc in enumerate(all_valid_counter_locations):
            all_features["counter{}_state".format(i)] = np.eye(N_COUNTER_STATES)[counter_location_to_state[c_loc]]

        for i, p_loc in enumerate(pot_locations):
            all_features["pot{}_state".format(i)] = np.eye(N_POT_STATES)[pot_tracker[p_loc]['state']]

        features_np = {k: np.array(v) for k, v in all_features.items()}

        # p0, p1 = overcooked_state.players
        # p0_dict = {k: v for k, v in features_np.items() if k[:2] == "p0"}
        # p1_dict = {k: v for k, v in features_np.items() if k[:2] == "p1"}
        # p0_features = np.concatenate(list(p0_dict.values()))
        # p1_features = np.concatenate(list(p1_dict.values()))

        # p1_rel_to_p0 = np.array(pos_distance(p1.position, p0.position))
        # abs_pos_p0 = np.array(p0.position)
        # ordered_features_p0 = np.squeeze(np.concatenate([p0_features, p1_features, p1_rel_to_p0, abs_pos_p0]))
        #
        # p0_rel_to_p1 = np.array(pos_distance(p0.position, p1.position))
        # abs_pos_p1 = np.array(p0.position)
        # ordered_features_p1 = np.squeeze(np.concatenate([p1_features, p0_features, p0_rel_to_p1, abs_pos_p1]))
        team_features = np.squeeze(np.concatenate(list(features_np.values())))
        return team_features

    def get_deltas_to_closest_location(self, player, locations, mlp):
        _, closest_loc = mlp.mp.min_cost_to_feature(player.pos_and_or, locations, with_argmin=True)
        if closest_loc is None:
            # "any object that does not exist or I am carrying is going to show up as a (0,0)
            # but I can disambiguate the two possibilities by looking at the features
            # for what kind of object I'm carrying"
            return (0, 0)
        dy_loc, dx_loc = pos_distance(closest_loc, player.position)
        return dy_loc, dx_loc

    ##############
    # DEPRECATED #
    ##############

    def calculate_distance_based_shaped_reward(self, state, new_state):
        """
        Adding reward shaping based on distance to certain features.
        """
        distance_based_shaped_reward = 0

        pot_states = self.get_pot_states(new_state)
        ready_pots = pot_states["tomato"]["ready"] + pot_states["onion"]["ready"]
        cooking_pots = ready_pots + pot_states["tomato"]["cooking"] + pot_states["onion"]["cooking"]
        nearly_ready_pots = cooking_pots + pot_states["tomato"]["partially_full"] + pot_states["onion"][
            "partially_full"]
        dishes_in_play = len(new_state.player_objects_by_type['dish'])
        for player_old, player_new in zip(state.players, new_state.players):
            # Linearly increase reward depending on vicinity to certain features, where distance of 10 achieves 0 reward
            max_dist = 8

            if player_new.held_object is not None and player_new.held_object.name == 'dish' and len(
                    nearly_ready_pots) >= dishes_in_play:
                min_dist_to_pot_new = np.inf
                min_dist_to_pot_old = np.inf
                for pot in nearly_ready_pots:
                    new_dist = np.linalg.norm(np.array(pot) - np.array(player_new.position))
                    old_dist = np.linalg.norm(np.array(pot) - np.array(player_old.position))
                    if new_dist < min_dist_to_pot_new:
                        min_dist_to_pot_new = new_dist
                    if old_dist < min_dist_to_pot_old:
                        min_dist_to_pot_old = old_dist
                if min_dist_to_pot_old > min_dist_to_pot_new:
                    distance_based_shaped_reward += self.reward_shaping_params["POT_DISTANCE_REW"] * (
                                1 - min(min_dist_to_pot_new / max_dist, 1))

            if player_new.held_object is None and len(cooking_pots) > 0 and dishes_in_play == 0:
                min_dist_to_d_new = np.inf
                min_dist_to_d_old = np.inf
                for serving_loc in self.terrain_pos_dict['D']:
                    new_dist = np.linalg.norm(np.array(serving_loc) - np.array(player_new.position))
                    old_dist = np.linalg.norm(np.array(serving_loc) - np.array(player_old.position))
                    if new_dist < min_dist_to_d_new:
                        min_dist_to_d_new = new_dist
                    if old_dist < min_dist_to_d_old:
                        min_dist_to_d_old = old_dist

                if min_dist_to_d_old > min_dist_to_d_new:
                    distance_based_shaped_reward += self.reward_shaping_params["DISH_DISP_DISTANCE_REW"] * (
                                1 - min(min_dist_to_d_new / max_dist, 1))

            if player_new.held_object is not None and player_new.held_object.name == 'soup':
                min_dist_to_s_new = np.inf
                min_dist_to_s_old = np.inf
                for serving_loc in self.terrain_pos_dict['S']:
                    new_dist = np.linalg.norm(np.array(serving_loc) - np.array(player_new.position))
                    old_dist = np.linalg.norm(np.array(serving_loc) - np.array(player_old.position))
                    if new_dist < min_dist_to_s_new:
                        min_dist_to_s_new = new_dist

                    if old_dist < min_dist_to_s_old:
                        min_dist_to_s_old = old_dist

                if min_dist_to_s_old > min_dist_to_s_new:
                    distance_based_shaped_reward += self.reward_shaping_params["SOUP_DISTANCE_REW"] * (
                                1 - min(min_dist_to_s_new / max_dist, 1))

        return distance_based_shaped_reward


# class OvercookedGridworld_Original(object):
#     """An MDP grid world based off of the Overcooked game."""
#     ORDER_TYPES = ObjectState.SOUP_TYPES + ['any']
#
#     def __init__(self, terrain, start_player_positions, start_order_list=None, cook_time=20, num_items_for_soup=3,
#                  delivery_reward=20, rew_shaping_params=None, layout_name="unnamed_layout"):
#         """
#         terrain: a matrix of strings that encode the MDP layout
#         layout_name: string identifier of the layout
#         start_player_positions: tuple of positions for both players' starting positions
#         start_order_list: either a tuple of orders or None if there is not specific list
#         cook_time: amount of timesteps required for a soup to cook
#         delivery_reward: amount of reward given per delivery
#         rew_shaping_params: reward given for completion of specific subgoals
#         """
#         self.height = len(terrain)
#         self.width = len(terrain[0])
#         self.shape = (self.width, self.height)
#         self.terrain_mtx = terrain
#         self.terrain_pos_dict = self._get_terrain_type_pos_dict()
#         self.start_player_positions = start_player_positions
#         self.num_players = len(start_player_positions)
#         self.start_order_list = start_order_list
#         self.soup_cooking_time = cook_time
#         self.num_items_for_soup = num_items_for_soup
#         self.delivery_reward = delivery_reward
#         self.reward_shaping_params = NO_REW_SHAPING_PARAMS if rew_shaping_params is None else rew_shaping_params
#         self.layout_name = layout_name
#
#     def __eq__(self, other):
#         return np.array_equal(self.terrain_mtx, other.terrain_mtx) and \
#                self.start_player_positions == other.start_player_positions and \
#                self.start_order_list == other.start_order_list and \
#                self.soup_cooking_time == other.soup_cooking_time and \
#                self.num_items_for_soup == other.num_items_for_soup and \
#                self.delivery_reward == other.delivery_reward and \
#                self.reward_shaping_params == other.reward_shaping_params and \
#                self.layout_name == other.layout_name
#
#     def copy(self):
#         return OvercookedGridworld(
#             terrain=self.terrain_mtx.copy(),
#             start_player_positions=self.start_player_positions,
#             start_order_list=None if self.start_order_list is None else list(self.start_order_list),
#             cook_time=self.soup_cooking_time,
#             num_items_for_soup=self.num_items_for_soup,
#             delivery_reward=self.delivery_reward,
#             rew_shaping_params=copy.deepcopy(self.reward_shaping_params),
#             layout_name=self.layout_name
#         )
#
#     @property
#     def mdp_params(self):
#         return {
#             "layout_name": self.layout_name,
#             "terrain": self.terrain_mtx,
#             "start_player_positions": self.start_player_positions,
#             "start_order_list": self.start_order_list,
#             "cook_time": self.soup_cooking_time,
#             "num_items_for_soup": self.num_items_for_soup,
#             "delivery_reward": self.delivery_reward,
#             "rew_shaping_params": copy.deepcopy(self.reward_shaping_params)
#         }
#
#     @staticmethod
#     def from_layout_name(layout_name, **params_to_overwrite):
#         """
#         Generates a OvercookedGridworld instance from a layout file.
#
#         One can overwrite the default mdp configuration using partial_mdp_config.
#         """
#         params_to_overwrite = params_to_overwrite.copy()
#         base_layout_params = read_layout_dict(layout_name)
#
#         grid = base_layout_params['grid']
#         del base_layout_params['grid']
#         base_layout_params['layout_name'] = layout_name
#
#         # Clean grid
#         grid = [layout_row.strip() for layout_row in grid.split("\n")]
#         return OvercookedGridworld.from_grid(grid, base_layout_params, params_to_overwrite)
#
#     @staticmethod
#     def from_grid(layout_grid, base_layout_params={}, params_to_overwrite={}, debug=False):
#         """
#         Returns instance of OvercookedGridworld with terrain and starting
#         positions derived from layout_grid.
#         One can override default configuration parameters of the mdp in
#         partial_mdp_config.
#         """
#         mdp_config = base_layout_params.copy()
#
#         layout_grid = [[c for c in row] for row in layout_grid]
#         OvercookedGridworld._assert_valid_grid(layout_grid)
#
#         player_positions = [None] * 9
#         for y, row in enumerate(layout_grid):
#             for x, c in enumerate(row):
#                 if c in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
#                     layout_grid[y][x] = ' '
#
#                     # -1 is to account for fact that player indexing starts from 1 rather than 0
#                     assert player_positions[int(c) - 1] is None, 'Duplicate player in grid'
#                     player_positions[int(c) - 1] = (x, y)
#
#         num_players = len([x for x in player_positions if x is not None])
#         player_positions = player_positions[:num_players]
#
#         # After removing player positions from grid we have a terrain mtx
#         mdp_config["terrain"] = layout_grid
#         mdp_config["start_player_positions"] = player_positions
#
#         for k, v in params_to_overwrite.items():
#             curr_val = mdp_config[k]
#             if debug:
#                 print("Overwriting mdp layout standard config value {}:{} -> {}".format(k, curr_val, v))
#             mdp_config[k] = v
#
#         return OvercookedGridworld(**mdp_config)
#
#     def get_actions(self, state):
#         """
#         Returns the list of lists of valid actions for 'state'.
#
#         The ith element of the list is the list of valid actions that player i
#         can take.
#         """
#         self._check_valid_state(state)
#         return [self._get_player_actions(state, i) for i in range(len(state.players))]
#
#     def _get_player_actions(self, state, player_num):
#         """All actions are allowed to all players in all states."""
#         return Action.ALL_ACTIONS
#
#     def _check_action(self, state, joint_action):
#         for p_action, p_legal_actions in zip(joint_action, self.get_actions(state)):
#             if p_action not in p_legal_actions:
#                 raise ValueError('Invalid action')
#
#     def get_standard_start_state(self):
#         start_state = OvercookedState.from_player_positions(
#             self.start_player_positions, order_list=self.start_order_list
#         )
#         return start_state
#
#     def get_random_start_state_fn(self, random_start_pos=False, rnd_obj_prob_thresh=0.0):
#         def start_state_fn():
#             if random_start_pos:
#                 valid_positions = self.get_valid_joint_player_positions()
#                 start_pos = valid_positions[np.random.choice(len(valid_positions))]
#             else:
#                 start_pos = self.start_player_positions
#
#             start_state = OvercookedState.from_player_positions(start_pos, order_list=self.start_order_list)
#
#             if rnd_obj_prob_thresh == 0:
#                 return start_state
#
#             # Arbitrary hard-coding for randomization of objects
#             # For each pot, add a random amount of onions with prob rnd_obj_prob_thresh
#             pots = self.get_pot_states(start_state)["empty"]
#             for pot_loc in pots:
#                 p = np.random.rand()
#                 if p < rnd_obj_prob_thresh:
#                     n = int(np.random.randint(low=1, high=4))
#                     start_state.objects[pot_loc] = ObjectState("soup", pot_loc, ('onion', n, 0))
#
#             # For each player, add a random object with prob rnd_obj_prob_thresh
#             for player in start_state.players:
#                 p = np.random.rand()
#                 if p < rnd_obj_prob_thresh:
#                     # Different objects have different probabilities
#                     obj = np.random.choice(["dish", "onion", "soup"], p=[0.2, 0.6, 0.2])
#                     if obj == "soup":
#                         player.set_object(
#                             ObjectState(obj, player.position,
#                                         ('onion', self.num_items_for_soup, self.soup_cooking_time))
#                         )
#                     else:
#                         player.set_object(ObjectState(obj, player.position))
#             return start_state
#
#         return start_state_fn
#
#     def is_terminal(self, state):
#         # There is a finite horizon, handled by the environment.
#         if state.order_list is None:
#             return False
#         return len(state.order_list) == 0
#
#     def get_valid_player_positions(self):
#         return self.terrain_pos_dict[' ']
#
#     def get_valid_joint_player_positions(self):
#         """Returns all valid tuples of the form (p0_pos, p1_pos, p2_pos, ...)"""
#         valid_positions = self.get_valid_player_positions()
#         all_joint_positions = list(itertools.product(valid_positions, repeat=self.num_players))
#         valid_joint_positions = [j_pos for j_pos in all_joint_positions if not self.is_joint_position_collision(j_pos)]
#         return valid_joint_positions
#
#     def get_valid_player_positions_and_orientations(self):
#         valid_states = []
#         for pos in self.get_valid_player_positions():
#             valid_states.extend([(pos, d) for d in Direction.ALL_DIRECTIONS])
#         return valid_states
#
#     def get_valid_joint_player_positions_and_orientations(self):
#         """All joint player position and orientation pairs that are not
#         overlapping and on empty terrain."""
#         valid_player_states = self.get_valid_player_positions_and_orientations()
#
#         valid_joint_player_states = []
#         for players_pos_and_orientations in itertools.product(valid_player_states, repeat=self.num_players):
#             joint_position = [plyer_pos_and_or[0] for plyer_pos_and_or in players_pos_and_orientations]
#             if not self.is_joint_position_collision(joint_position):
#                 valid_joint_player_states.append(players_pos_and_orientations)
#
#         return valid_joint_player_states
#
#     def get_adjacent_features(self, player):
#         adj_feats = []
#         pos = player.position
#         for d in Direction.ALL_DIRECTIONS:
#             adj_pos = Action.move_in_direction(pos, d)
#             adj_feats.append((pos, self.get_terrain_type_at_pos(adj_pos)))
#         return adj_feats
#
#     def get_terrain_type_at_pos(self, pos):
#         x, y = pos
#         return self.terrain_mtx[y][x]
#
#     def get_dish_dispenser_locations(self):
#         return list(self.terrain_pos_dict['D'])
#
#     def get_onion_dispenser_locations(self):
#         return list(self.terrain_pos_dict['O'])
#
#     def get_tomato_dispenser_locations(self):
#         return list(self.terrain_pos_dict['T'])
#
#     def get_serving_locations(self):
#         return list(self.terrain_pos_dict['S'])
#
#     def get_pot_locations(self):
#         return list(self.terrain_pos_dict['P'])
#
#     def get_counter_locations(self):
#         return list(self.terrain_pos_dict['X'])
#
#     def get_pot_states(self, state):
#         """Returns dict with structure:
#         {
#          empty: [ObjStates]
#          onion: {
#             'x_items': [soup objects with x items],
#             'cooking': [ready soup objs]
#             'ready': [ready soup objs],
#             'partially_full': [all non-empty and non-full soups]
#             }
#          tomato: same dict structure as above
#         }
#         """
#         pots_states_dict = {}
#         pots_states_dict['empty'] = []
#         pots_states_dict['onion'] = defaultdict(list)
#         pots_states_dict['tomato'] = defaultdict(list)
#         for pot_pos in self.get_pot_locations():
#             if not state.has_object(pot_pos):
#                 pots_states_dict['empty'].append(pot_pos)
#             else:
#                 soup_obj = state.get_object(pot_pos)
#                 soup_type, num_items, cook_time = soup_obj.state
#                 if 0 < num_items < self.num_items_for_soup:
#                     pots_states_dict[soup_type]['{}_items'.format(num_items)].append(pot_pos)
#                 elif num_items == self.num_items_for_soup:
#                     assert cook_time <= self.soup_cooking_time
#                     if cook_time == self.soup_cooking_time:
#                         pots_states_dict[soup_type]['ready'].append(pot_pos)
#                     else:
#                         pots_states_dict[soup_type]['cooking'].append(pot_pos)
#                 else:
#                     raise ValueError("Pot with more than {} items".format(self.num_items_for_soup))
#
#                 if 0 < num_items < self.num_items_for_soup:
#                     pots_states_dict[soup_type]['partially_full'].append(pot_pos)
#
#         return pots_states_dict
#
#     def get_counter_objects_dict(self, state, counter_subset=None):
#         """Returns a dictionary of pos:objects on counters by type"""
#         counters_considered = self.terrain_pos_dict['X'] if counter_subset is None else counter_subset
#         counter_objects_dict = defaultdict(list)
#         for obj in state.objects.values():
#             if obj.position in counters_considered:
#                 counter_objects_dict[obj.name].append(obj.position)
#         return counter_objects_dict
#
#     def get_empty_counter_locations(self, state):
#         counter_locations = self.get_counter_locations()
#         return [pos for pos in counter_locations if not state.has_object(pos)]
#
#     def get_state_transition(self, state, joint_action):
#         """Gets information about possible transitions for the action.
#
#         Returns the next state, sparse reward and reward shaping.
#         Assumes all actions are deterministic.
#
#         NOTE: Sparse reward is given only when soups are delivered,
#         shaped reward is given only for completion of subgoals
#         (not soup deliveries).
#         """
#         assert not self.is_terminal(state), "Trying to find successor of a terminal state: {}".format(state)
#         for action, action_set in zip(joint_action, self.get_actions(state)):
#             if action not in action_set:
#                 raise ValueError("Illegal action %s in state %s" % (action, state))
#
#         new_state = state.deepcopy()
#
#         # Resolve interacts first
#         sparse_reward, shaped_reward = self.resolve_interacts(new_state, joint_action)
#
#         assert new_state.player_positions == state.player_positions
#         assert new_state.player_orientations == state.player_orientations
#
#         # Resolve player movements
#         self.resolve_movement(new_state, joint_action)
#
#         # Finally, environment effects
#         sparse_reward += self.step_environment_effects(new_state)
#
#         # Additional dense reward logic
#         # shaped_reward += self.calculate_distance_based_shaped_reward(state, new_state)
#
#         return new_state, sparse_reward, shaped_reward
#
#     def resolve_interacts(self, new_state, joint_action):
#         """
#         Resolve any INTERACT actions, if present.
#
#         Currently if two players both interact with a terrain, we resolve player 1's interact
#         first and then player 2's, without doing anything like collision checking.
#         """
#         pot_states = self.get_pot_states(new_state)
#         ready_pots = pot_states["tomato"]["ready"] + pot_states["onion"]["ready"]
#         cooking_pots = ready_pots + pot_states["tomato"]["cooking"] + pot_states["onion"]["cooking"]
#         nearly_ready_pots = cooking_pots + pot_states["tomato"]["partially_full"] + pot_states["onion"][
#             "partially_full"]
#
#         sparse_reward, shaped_reward = 0, 0
#         for player, action in zip(new_state.players, joint_action):
#             if action != Action.INTERACT:
#                 continue
#
#             pos, o = player.position, player.orientation
#             i_pos = Action.move_in_direction(pos, o)
#             terrain_type = self.get_terrain_type_at_pos(i_pos)
#
#             if terrain_type == 'X':
#                 if player.has_object() and not new_state.has_object(i_pos):
#                     new_state.add_object(player.remove_object(), i_pos)
#                 elif not player.has_object() and new_state.has_object(i_pos):
#                     player.set_object(new_state.remove_object(i_pos))
#
#             elif terrain_type == 'O' and player.held_object is None:
#                 player.set_object(ObjectState('onion', pos))
#             elif terrain_type == 'T' and player.held_object is None:
#                 player.set_object(ObjectState('tomato', pos))
#             elif terrain_type == 'D' and player.held_object is None:
#                 dishes_already = len(new_state.player_objects_by_type['dish'])
#                 player.set_object(ObjectState('dish', pos))
#
#                 dishes_on_counters = self.get_counter_objects_dict(new_state)["dish"]
#                 if len(nearly_ready_pots) > dishes_already and len(dishes_on_counters) == 0:
#                     shaped_reward += self.reward_shaping_params["DISH_PICKUP_REWARD"]
#
#             elif terrain_type == 'P' and player.has_object():
#                 if player.get_object().name == 'dish' and new_state.has_object(i_pos):
#                     obj = new_state.get_object(i_pos)
#                     assert obj.name == 'soup', 'Object in pot was not soup'
#                     _, num_items, cook_time = obj.state
#                     if num_items == self.num_items_for_soup and cook_time >= self.soup_cooking_time:
#                         player.remove_object()  # Turn the dish into the soup
#                         player.set_object(new_state.remove_object(i_pos))
#                         shaped_reward += self.reward_shaping_params["SOUP_PICKUP_REWARD"]
#
#                 elif player.get_object().name in ['onion', 'tomato']:
#                     item_type = player.get_object().name
#
#                     if not new_state.has_object(i_pos):
#                         # Pot was empty
#                         player.remove_object()
#                         new_state.add_object(ObjectState('soup', i_pos, (item_type, 1, 0)), i_pos)
#                         shaped_reward += self.reward_shaping_params["PLACEMENT_IN_POT_REW"]
#
#                     else:
#                         # Pot has already items in it
#                         obj = new_state.get_object(i_pos)
#                         assert obj.name == 'soup', 'Object in pot was not soup'
#                         soup_type, num_items, cook_time = obj.state
#                         if num_items < self.num_items_for_soup and soup_type == item_type:
#                             player.remove_object()
#                             obj.state = (soup_type, num_items + 1, 0)
#                             shaped_reward += self.reward_shaping_params["PLACEMENT_IN_POT_REW"]
#
#             elif terrain_type == 'S' and player.has_object():
#                 obj = player.get_object()
#                 if obj.name == 'soup':
#
#                     new_state, delivery_rew = self.deliver_soup(new_state, player, obj)
#                     sparse_reward += delivery_rew
#
#                     # If last soup necessary was delivered, stop resolving interacts
#                     if new_state.order_list is not None and len(new_state.order_list) == 0:
#                         break
#
#         return sparse_reward, shaped_reward
#
#     def deliver_soup(self, state, player, soup_obj):
#         """
#         Deliver the soup, and get reward if there is no order list
#         or if the type of the delivered soup matches the next order.
#         """
#         soup_type, num_items, cook_time = soup_obj.state
#         assert soup_type in ObjectState.SOUP_TYPES
#         assert num_items == self.num_items_for_soup
#         assert cook_time >= self.soup_cooking_time, "Cook time {} mdp cook time {}".format(cook_time,
#                                                                                            self.soup_cooking_time)
#         player.remove_object()
#
#         if state.order_list is None:
#             return state, self.delivery_reward
#
#         # If the delivered soup is the one currently required
#         assert not self.is_terminal(state)
#         current_order = state.order_list[0]
#         if current_order == 'any' or soup_type == current_order:
#             state.order_list = state.order_list[1:]
#             return state, self.delivery_reward
#
#         return state, 0
#
#     def resolve_movement(self, state, joint_action):
#         """Resolve player movement and deal with possible collisions"""
#         new_positions, new_orientations = self.compute_new_positions_and_orientations(state.players, joint_action)
#         for player_state, new_pos, new_o in zip(state.players, new_positions, new_orientations):
#             player_state.update_pos_and_or(new_pos, new_o)
#
#     def compute_new_positions_and_orientations(self, old_player_states, joint_action):
#         """Compute new positions and orientations ignoring collisions"""
#         new_positions, new_orientations = list(zip(*[
#             self._move_if_direction(p.position, p.orientation, a) \
#             for p, a in zip(old_player_states, joint_action)]))
#         old_positions = tuple(p.position for p in old_player_states)
#         new_positions, collided = self._handle_collisions(old_positions, new_positions)
#         return new_positions, new_orientations
#
#     def is_transition_collision(self, old_positions, new_positions):
#         # Checking for any players ending in same square
#         if self.is_joint_position_collision(new_positions):
#             return True
#         # Check if any two players crossed paths
#         for idx0, idx1 in itertools.combinations(range(self.num_players), 2):
#             p1_old, p2_old = old_positions[idx0], old_positions[idx1]
#             p1_new, p2_new = new_positions[idx0], new_positions[idx1]
#             if p1_new == p2_old and p1_old == p2_new:
#                 return True
#         return False
#
#     def is_joint_position_collision(self, joint_position):
#         return any(pos0 == pos1 for pos0, pos1 in itertools.combinations(joint_position, 2))
#
#     def step_environment_effects(self, state):
#         reward = 0
#         for obj in state.objects.values():
#             if obj.name == 'soup':
#                 x, y = obj.position
#                 soup_type, num_items, cook_time = obj.state
#                 # NOTE: cook_time is capped at self.soup_cooking_time
#                 if self.terrain_mtx[y][x] == 'P' and \
#                         num_items == self.num_items_for_soup and \
#                         cook_time < self.soup_cooking_time:
#                     obj.state = soup_type, num_items, cook_time + 1
#         return reward
#
#     def _handle_collisions(self, old_positions, new_positions):
#         """If agents collide, they stay at their old locations"""
#         if self.is_transition_collision(old_positions, new_positions):
#             return old_positions
#         return new_positions
#
#     def _get_terrain_type_pos_dict(self):
#         pos_dict = defaultdict(list)
#         for y, terrain_row in enumerate(self.terrain_mtx):
#             for x, terrain_type in enumerate(terrain_row):
#                 pos_dict[terrain_type].append((x, y))
#         return pos_dict
#
#     def _move_if_direction(self, position, orientation, action):
#         """Returns position and orientation that would
#         be obtained after executing action"""
#         if action == Action.INTERACT:
#             return position, orientation
#         new_pos = Action.move_in_direction(position, action)
#         new_orientation = orientation if action == Action.STAY else action
#         if new_pos not in self.get_valid_player_positions():
#             return position, new_orientation
#         return new_pos, new_orientation
#
#     def _check_valid_state(self, state):
#         """Checks that the state is valid.
#
#         Conditions checked:
#         - Players are on free spaces, not terrain
#         - Held objects have the same position as the player holding them
#         - Non-held objects are on terrain
#         - No two players or non-held objects occupy the same position
#         - Objects have a valid state (eg. no pot with 4 onions)
#         """
#         all_objects = list(state.objects.values())
#         for player_state in state.players:
#             # Check that players are not on terrain
#             pos = player_state.position
#             assert pos in self.get_valid_player_positions()
#
#             # Check that held objects have the same position
#             if player_state.held_object is not None:
#                 all_objects.append(player_state.held_object)
#                 assert player_state.held_object.position == player_state.position
#
#         for obj_pos, obj_state in state.objects.items():
#             # Check that the hash key position agrees with the position stored
#             # in the object state
#             assert obj_state.position == obj_pos
#             # Check that non-held objects are on terrain
#             assert self.get_terrain_type_at_pos(obj_pos) != ' '
#
#         # Check that players and non-held objects don't overlap
#         all_pos = [player_state.position for player_state in state.players]
#         all_pos += [obj_state.position for obj_state in state.objects.values()]
#         assert len(all_pos) == len(set(all_pos)), "Overlapping players or objects"
#
#         # Check that objects have a valid state
#         for obj_state in all_objects:
#             assert obj_state.is_valid()
#
#     @staticmethod
#     def _assert_valid_grid(grid):
#         """Raises an AssertionError if the grid is invalid.
#
#         grid:  A sequence of sequences of spaces, representing a grid of a
#         certain height and width. grid[y][x] is the space at row y and column
#         x. A space must be either 'X' (representing a counter), ' ' (an empty
#         space), 'O' (onion supply), 'P' (pot), 'D' (dish supply), 'S' (serving
#         location), '1' (player 1) and '2' (player 2).
#         """
#         height = len(grid)
#         width = len(grid[0])
#
#         # Make sure the grid is not ragged
#         assert all(len(row) == width for row in grid), 'Ragged grid'
#
#         # Borders must not be free spaces
#         def is_not_free(c):
#             return c in 'XOPDST'
#
#         for y in range(height):
#             assert is_not_free(grid[y][0]), 'Left border must not be free'
#             assert is_not_free(grid[y][-1]), 'Right border must not be free'
#         for x in range(width):
#             assert is_not_free(grid[0][x]), 'Top border must not be free'
#             assert is_not_free(grid[-1][x]), 'Bottom border must not be free'
#
#         all_elements = [element for row in grid for element in row]
#         digits = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
#         layout_digits = [e for e in all_elements if e in digits]
#         num_players = len(layout_digits)
#         assert num_players > 0, "No players (digits) in grid"
#         layout_digits = list(sorted(map(int, layout_digits)))
#         assert layout_digits == list(range(1, num_players + 1)), "Some players were missing"
#
#         assert all(c in 'XOPDST123456789 ' for c in all_elements), 'Invalid character in grid'
#         assert all_elements.count('1') == 1, "'1' must be present exactly once"
#         assert all_elements.count('D') >= 1, "'D' must be present at least once"
#         assert all_elements.count('S') >= 1, "'S' must be present at least once"
#         assert all_elements.count('P') >= 1, "'P' must be present at least once"
#         assert all_elements.count('O') >= 1 or all_elements.count('T') >= 1, "'O' or 'T' must be present at least once"
#
#     #####################
#     # TERMINAL GRAPHICS #
#     #####################
#
#     def state_string(self, state):
#         """String representation of the current state"""
#         players_dict = {player.position: player for player in state.players}
#
#         grid_string = ""
#         for y, terrain_row in enumerate(self.terrain_mtx):
#             for x, element in enumerate(terrain_row):
#                 if (x, y) in players_dict.keys():
#                     player = players_dict[(x, y)]
#                     orientation = player.orientation
#                     assert orientation in Direction.ALL_DIRECTIONS
#
#                     grid_string += Action.ACTION_TO_CHAR[orientation]
#                     player_object = player.held_object
#                     if player_object:
#                         grid_string += player_object.name[:1]
#                     else:
#                         player_idx_lst = [i for i, p in enumerate(state.players) if p.position == player.position]
#                         assert len(player_idx_lst) == 1
#                         grid_string += str(player_idx_lst[0])
#                 else:
#                     if element == "X" and state.has_object((x, y)):
#                         state_obj = state.get_object((x, y))
#                         grid_string = grid_string + element + state_obj.name[:1]
#
#                     elif element == "P" and state.has_object((x, y)):
#                         soup_obj = state.get_object((x, y))
#                         soup_type, num_items, cook_time = soup_obj.state
#                         if soup_type == "onion":
#                             grid_string += "ø"
#                         elif soup_type == "tomato":
#                             grid_string += "†"
#                         else:
#                             raise ValueError()
#
#                         if num_items == self.num_items_for_soup:
#                             grid_string += str(cook_time)
#
#                         # NOTE: do not currently have terminal graphics
#                         # support for cooking times greater than 3.
#                         elif num_items == 2:
#                             grid_string += "="
#                         else:
#                             grid_string += "-"
#                     else:
#                         grid_string += element + " "
#
#             grid_string += "\n"
#
#         if state.order_list is not None:
#             grid_string += "Current orders: {}/{} are any's\n".format(
#                 len(state.order_list), len([order == "any" for order in state.order_list])
#             )
#         return grid_string
#
#     ###################
#     # STATE ENCODINGS #
#     ###################
#
#     def lossless_state_encoding(self, overcooked_state, debug=False):
#         """Featurizes a OvercookedState object into a stack of boolean masks that are easily readable by a CNN"""
#         assert type(debug) is bool
#         base_map_features = ["pot_loc", "counter_loc", "onion_disp_loc", "dish_disp_loc", "serve_loc"]
#         variable_map_features = ["onions_in_pot", "onions_cook_time", "onion_soup_loc", "dishes", "onions"]
#
#         all_objects = overcooked_state.all_objects_list
#
#         def make_layer(position, value):
#             layer = np.zeros(self.shape)
#             layer[position] = value
#             return layer
#
#         def process_for_player(primary_agent_idx):
#             # Ensure that primary_agent_idx layers are ordered before other_agent_idx layers
#             other_agent_idx = 1 - primary_agent_idx
#             ordered_player_features = ["player_{}_loc".format(primary_agent_idx),
#                                        "player_{}_loc".format(other_agent_idx)] + \
#                                       ["player_{}_orientation_{}".format(i, Direction.DIRECTION_TO_INDEX[d])
#                                        for i, d in itertools.product([primary_agent_idx, other_agent_idx],
#                                                                      Direction.ALL_DIRECTIONS)]
#
#             LAYERS = ordered_player_features + base_map_features + variable_map_features
#             state_mask_dict = {k: np.zeros(self.shape) for k in LAYERS}
#
#             # MAP LAYERS
#             for loc in self.get_counter_locations():
#                 state_mask_dict["counter_loc"][loc] = 1
#
#             for loc in self.get_pot_locations():
#                 state_mask_dict["pot_loc"][loc] = 1
#
#             for loc in self.get_onion_dispenser_locations():
#                 state_mask_dict["onion_disp_loc"][loc] = 1
#
#             for loc in self.get_dish_dispenser_locations():
#                 state_mask_dict["dish_disp_loc"][loc] = 1
#
#             for loc in self.get_serving_locations():
#                 state_mask_dict["serve_loc"][loc] = 1
#
#             # PLAYER LAYERS
#             for i, player in enumerate(overcooked_state.players):
#                 player_orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
#                 state_mask_dict["player_{}_loc".format(i)] = make_layer(player.position, 1)
#                 state_mask_dict["player_{}_orientation_{}".format(i, player_orientation_idx)] = make_layer(
#                     player.position, 1)
#
#             # OBJECT & STATE LAYERS
#             for obj in all_objects:
#                 if obj.name == "soup":
#                     soup_type, num_onions, cook_time = obj.state
#                     if soup_type == "onion":
#                         if obj.position in self.get_pot_locations():
#                             soup_type, num_onions, cook_time = obj.state
#                             state_mask_dict["onions_in_pot"] += make_layer(obj.position, num_onions)
#                             state_mask_dict["onions_cook_time"] += make_layer(obj.position, cook_time)
#                         else:
#                             # If player soup is not in a pot, put it in separate mask
#                             state_mask_dict["onion_soup_loc"] += make_layer(obj.position, 1)
#                     else:
#                         raise ValueError("Unrecognized soup")
#
#                 elif obj.name == "dish":
#                     state_mask_dict["dishes"] += make_layer(obj.position, 1)
#                 elif obj.name == "onion":
#                     state_mask_dict["onions"] += make_layer(obj.position, 1)
#                 else:
#                     raise ValueError("Unrecognized object")
#
#             if debug:
#                 print(len(LAYERS))
#                 print(len(state_mask_dict))
#                 for k, v in state_mask_dict.items():
#                     print(k)
#                     print(np.transpose(v, (1, 0)))
#
#             # Stack of all the state masks, order decided by order of LAYERS
#             state_mask_stack = np.array([state_mask_dict[layer_id] for layer_id in LAYERS])
#             state_mask_stack = np.transpose(state_mask_stack, (1, 2, 0))
#             assert state_mask_stack.shape[:2] == self.shape
#             assert state_mask_stack.shape[2] == len(LAYERS)
#             # NOTE: currently not including time left or order_list in featurization
#             return np.array(state_mask_stack).astype(int)
#
#         # NOTE: Currently not very efficient, a decent amount of computation repeated here
#         num_players = len(overcooked_state.players)
#         final_obs_for_players = tuple(process_for_player(i) for i in range(num_players))
#         return final_obs_for_players
#
#     def featurize_state(self, overcooked_state, mlp):
#         """
#         Encode state with some manually designed features.
#         NOTE: currently works for just two players.
#         """
#
#         all_features = {}
#
#         def make_closest_feature(idx, name, locations):
#             "Compute (x, y) deltas to closest feature of type `name`, and save it in the features dict"
#             all_features["p{}_closest_{}".format(idx, name)] = self.get_deltas_to_closest_location(player, locations,
#                                                                                                    mlp)
#
#         IDX_TO_OBJ = ["onion", "soup", "dish"]
#         OBJ_TO_IDX = {o_name: idx for idx, o_name in enumerate(IDX_TO_OBJ)}
#
#         counter_objects = self.get_counter_objects_dict(overcooked_state)
#         pot_state = self.get_pot_states(overcooked_state)
#
#         # Player Info
#         for i, player in enumerate(overcooked_state.players):
#             orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
#             all_features["p{}_orientation".format(i)] = np.eye(4)[orientation_idx]
#             obj = player.held_object
#
#             if obj is None:
#                 held_obj_name = "none"
#                 all_features["p{}_objs".format(i)] = np.zeros(len(IDX_TO_OBJ))
#             else:
#                 held_obj_name = obj.name
#                 obj_idx = OBJ_TO_IDX[held_obj_name]
#                 all_features["p{}_objs".format(i)] = np.eye(len(IDX_TO_OBJ))[obj_idx]
#
#             # Closest feature of each type
#             if held_obj_name == "onion":
#                 all_features["p{}_closest_onion".format(i)] = (0, 0)
#             else:
#                 make_closest_feature(i, "onion", self.get_onion_dispenser_locations() + counter_objects["onion"])
#
#             make_closest_feature(i, "empty_pot", pot_state["empty"])
#             make_closest_feature(i, "one_onion_pot", pot_state["onion"]["one_onion"])
#             make_closest_feature(i, "two_onion_pot", pot_state["onion"]["two_onion"])
#             make_closest_feature(i, "cooking_pot", pot_state["onion"]["cooking"])
#             make_closest_feature(i, "ready_pot", pot_state["onion"]["ready"])
#
#             if held_obj_name == "dish":
#                 all_features["p{}_closest_dish".format(i)] = (0, 0)
#             else:
#                 make_closest_feature(i, "dish", self.get_dish_dispenser_locations() + counter_objects["dish"])
#
#             if held_obj_name == "soup":
#                 all_features["p{}_closest_soup".format(i)] = (0, 0)
#             else:
#                 make_closest_feature(i, "soup", counter_objects["soup"])
#
#             make_closest_feature(i, "serving", self.get_serving_locations())
#
#             for direction, pos_and_feat in enumerate(self.get_adjacent_features(player)):
#                 adj_pos, feat = pos_and_feat
#
#                 if direction == player.orientation:
#                     # Check if counter we are facing is empty
#                     facing_counter = (feat == 'X' and adj_pos not in overcooked_state.objects.keys())
#                     facing_counter_feature = [1] if facing_counter else [0]
#                     all_features["p{}_facing_empty_counter".format(i)] = facing_counter_feature
#
#                 all_features["p{}_wall_{}".format(i, direction)] = [0] if feat == ' ' else [1]
#
#         features_np = {k: np.array(v) for k, v in all_features.items()}
#
#         p0, p1 = overcooked_state.players
#         p0_dict = {k: v for k, v in features_np.items() if k[:2] == "p0"}
#         p1_dict = {k: v for k, v in features_np.items() if k[:2] == "p1"}
#         p0_features = np.concatenate(list(p0_dict.values()))
#         p1_features = np.concatenate(list(p1_dict.values()))
#
#         p1_rel_to_p0 = np.array(pos_distance(p1.position, p0.position))
#         abs_pos_p0 = np.array(p0.position)
#         ordered_features_p0 = np.squeeze(np.concatenate([p0_features, p1_features, p1_rel_to_p0, abs_pos_p0]))
#
#         p0_rel_to_p1 = np.array(pos_distance(p0.position, p1.position))
#         abs_pos_p1 = np.array(p0.position)
#         ordered_features_p1 = np.squeeze(np.concatenate([p1_features, p0_features, p0_rel_to_p1, abs_pos_p1]))
#         return ordered_features_p0, ordered_features_p1
#
#     def get_deltas_to_closest_location(self, player, locations, mlp):
#         _, closest_loc = mlp.mp.min_cost_to_feature(player.pos_and_or, locations, with_argmin=True)
#         if closest_loc is None:
#             # "any object that does not exist or I am carrying is going to show up as a (0,0)
#             # but I can disambiguate the two possibilities by looking at the features
#             # for what kind of object I'm carrying"
#             return (0, 0)
#         dy_loc, dx_loc = pos_distance(closest_loc, player.position)
#         return dy_loc, dx_loc
#
#     ##############
#     # DEPRECATED #
#     ##############
#
#     def calculate_distance_based_shaped_reward(self, state, new_state):
#         """
#         Adding reward shaping based on distance to certain features.
#         """
#         distance_based_shaped_reward = 0
#
#         pot_states = self.get_pot_states(new_state)
#         ready_pots = pot_states["tomato"]["ready"] + pot_states["onion"]["ready"]
#         cooking_pots = ready_pots + pot_states["tomato"]["cooking"] + pot_states["onion"]["cooking"]
#         nearly_ready_pots = cooking_pots + pot_states["tomato"]["partially_full"] + pot_states["onion"][
#             "partially_full"]
#         dishes_in_play = len(new_state.player_objects_by_type['dish'])
#         for player_old, player_new in zip(state.players, new_state.players):
#             # Linearly increase reward depending on vicinity to certain features, where distance of 10 achieves 0 reward
#             max_dist = 8
#
#             if player_new.held_object is not None and player_new.held_object.name == 'dish' and len(
#                     nearly_ready_pots) >= dishes_in_play:
#                 min_dist_to_pot_new = np.inf
#                 min_dist_to_pot_old = np.inf
#                 for pot in nearly_ready_pots:
#                     new_dist = np.linalg.norm(np.array(pot) - np.array(player_new.position))
#                     old_dist = np.linalg.norm(np.array(pot) - np.array(player_old.position))
#                     if new_dist < min_dist_to_pot_new:
#                         min_dist_to_pot_new = new_dist
#                     if old_dist < min_dist_to_pot_old:
#                         min_dist_to_pot_old = old_dist
#                 if min_dist_to_pot_old > min_dist_to_pot_new:
#                     distance_based_shaped_reward += self.reward_shaping_params["POT_DISTANCE_REW"] * (
#                                 1 - min(min_dist_to_pot_new / max_dist, 1))
#
#             if player_new.held_object is None and len(cooking_pots) > 0 and dishes_in_play == 0:
#                 min_dist_to_d_new = np.inf
#                 min_dist_to_d_old = np.inf
#                 for serving_loc in self.terrain_pos_dict['D']:
#                     new_dist = np.linalg.norm(np.array(serving_loc) - np.array(player_new.position))
#                     old_dist = np.linalg.norm(np.array(serving_loc) - np.array(player_old.position))
#                     if new_dist < min_dist_to_d_new:
#                         min_dist_to_d_new = new_dist
#                     if old_dist < min_dist_to_d_old:
#                         min_dist_to_d_old = old_dist
#
#                 if min_dist_to_d_old > min_dist_to_d_new:
#                     distance_based_shaped_reward += self.reward_shaping_params["DISH_DISP_DISTANCE_REW"] * (
#                                 1 - min(min_dist_to_d_new / max_dist, 1))
#
#             if player_new.held_object is not None and player_new.held_object.name == 'soup':
#                 min_dist_to_s_new = np.inf
#                 min_dist_to_s_old = np.inf
#                 for serving_loc in self.terrain_pos_dict['S']:
#                     new_dist = np.linalg.norm(np.array(serving_loc) - np.array(player_new.position))
#                     old_dist = np.linalg.norm(np.array(serving_loc) - np.array(player_old.position))
#                     if new_dist < min_dist_to_s_new:
#                         min_dist_to_s_new = new_dist
#
#                     if old_dist < min_dist_to_s_old:
#                         min_dist_to_s_old = old_dist
#
#                 if min_dist_to_s_old > min_dist_to_s_new:
#                     distance_based_shaped_reward += self.reward_shaping_params["SOUP_DISTANCE_REW"] * (
#                                 1 - min(min_dist_to_s_new / max_dist, 1))
#
#         return distance_based_shaped_reward