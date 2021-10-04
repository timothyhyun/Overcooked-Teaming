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


def test_random_agents():
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

    # print('results', random_results)
    avg_random_results = np.mean(random_results['ep_returns'])
    print('avg', avg_random_results)
    return random_results



if __name__ == "__main__":
    test_random_agents()







