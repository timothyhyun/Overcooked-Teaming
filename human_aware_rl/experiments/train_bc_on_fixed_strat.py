import copy
import numpy as np

import sys, os
import pickle

sys.path.insert(0, "../")
# print('path', os.path.dirname(os.path.abspath(__file__)))

from overcooked_ai_py.utils import save_pickle, mean_and_std_err
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import AgentPair

from human_aware_rl.utils import reset_tf, set_global_seed, common_keys_equal
from human_aware_rl.imitation.behavioral_cloning_fixed_fc import train_bc_agent, eval_with_benchmarking_from_saved, BC_SAVE_DIR, \
    plot_bc_run, \
    DEFAULT_BC_PARAMS, get_bc_agent_from_saved, plot_bc_run_modified

# Path for dict containing the best bc models paths
BEST_BC_MODELS_PATH = BC_SAVE_DIR + "best_bc_model_paths"
BC_MODELS_EVALUATION_PATH = BC_SAVE_DIR + "bc_models_all_evaluations"


def train_bc_agent_from_hh_data(layout_name, fixed_strat_params, agent_name, num_epochs, lr, adam_eps, model, finetuning=False, save_agents_name=''):
    """Trains a BC agent from human human data (model can either be `train` or `test`, which is trained
    on two different subsets of the data)."""

    bc_params = copy.deepcopy(DEFAULT_BC_PARAMS)
    bc_params["data_params"]['train_mdps'] = [layout_name]
    # bc_params["data_params"]['data_path'] = "../data/human/anonymized/clean_{}_trials.pkl".format(model)
    bc_params["mdp_params"]['layout_name'] = layout_name
    bc_params["mdp_params"]['start_order_list'] = None

    model_save_dir = f"bc_train_fixed_strat_{save_agents_name}_{layout_name}_{agent_name}/"
    # model_save_dir = layout_name + "_" + agent_name + "/"
    # train_bc_agent() returns the bc model
    # if model == 'train':
    #     is_train = True
    # else:
    #     is_train = False

    # if finetuning:
    #     return train_bc_agent_w_finetuning(model_save_dir, bc_params, num_epochs=num_epochs, lr=lr, adam_eps=adam_eps)
    # return train_bc_agent(model_save_dir, bc_params, num_epochs=num_epochs, lr=lr, adam_eps=adam_eps)
    return train_bc_agent(model_save_dir, bc_params, fixed_strat_params=fixed_strat_params, num_epochs=num_epochs, lr=lr, adam_eps=adam_eps)


def train_bc_models_on_fixed_agent_trajs(a0_type, a1_type, N_teams, params, seed=0):
    """ Train 1 model for a set of fixed agents for Forced Coordination """
    fixed_params = (a0_type, a1_type, N_teams)

    set_global_seed(seed)
    type_string = a0_type + '_' + a1_type
    # 1. Train BC model
    model = train_bc_agent_from_hh_data(agent_name="bc_train_seed{}".format(seed), fixed_strat_params=fixed_params, model='train', **params,
                                        finetuning=False, save_agents_name=type_string)
    # plot_bc_run(model.bc_info, params['num_epochs'])
    run_name = a0_type + '_'+a1_type
    plot_bc_run_modified(model.bc_info, params['num_epochs'], run_name, seed)

    # 2. Test BC model
    # model = train_bc_agent_from_hh_data(agent_name="bc_test_seed{}".format(seed), model='test', **params,
    #                                     finetuning=False)
    # # plot_bc_run(model.bc_info, params['num_epochs'])
    # plot_bc_run_modified(model.bc_info, params['num_epochs'], seed_idx, seed)
    # reset_tf()


def evaluate_all_bc_models(all_params, num_rounds, num_seeds):
    """Evaluate all trained models"""
    bc_models_evaluation = {}
    for params in all_params:
        layout_name = params["layout_name"]

        print(layout_name)
        bc_models_evaluation[layout_name] = {"train": {}, "test": {}}

        for seed_idx in range(num_seeds):
            # For all params and seeds, evaluate the model with a saved model. Pass in the BC model file name to evaluate.
            eval_trajs = eval_with_benchmarking_from_saved(num_rounds,
                                                           layout_name + "_bc_train_seed{}".format(seed_idx))
            bc_models_evaluation[layout_name]["train"][seed_idx] = np.mean(eval_trajs['ep_returns'])
            pickle.dump(eval_trajs, open('saved_eval_trajs/train_train_ex4_dual.pkl', 'wb'))

            eval_trajs = eval_with_benchmarking_from_saved(num_rounds, layout_name + "_bc_test_seed{}".format(seed_idx))
            bc_models_evaluation[layout_name]["test"][seed_idx] = np.mean(eval_trajs['ep_returns'])
            pickle.dump(eval_trajs, open('saved_eval_trajs/test_test_ex4_dual.pkl', 'wb'))

    return bc_models_evaluation


def evaluate_bc_models(bc_model_paths, num_rounds):
    """
    Evaluate BC models passed in over `num_rounds` rounds
    """
    best_bc_models_performance = {}

    # Evaluate best
    for layout_name in bc_model_paths['train'].keys():
        # For each layout
        print(layout_name)
        best_bc_models_performance[layout_name] = {}

        # Evaluate the BC train model playing with itself. Get mean and stderr of the returns of rollouts. This is "BC_train+BC_train".
        eval_trajs = eval_with_benchmarking_from_saved(num_rounds, bc_model_paths['train'][layout_name])
        best_bc_models_performance[layout_name]["BC_train+BC_train"] = mean_and_std_err(eval_trajs['ep_returns'])
        pickle.dump(eval_trajs, open('saved_eval_trajs/bc_train_and_bc_train_ex4_dual.pkl', 'wb'))

        # Evaluate the BC test model playing with itself. Get mean and stderr of the returns of rollouts. This is "BC_test+BC_test".
        eval_trajs = eval_with_benchmarking_from_saved(num_rounds, bc_model_paths['test'][layout_name])
        best_bc_models_performance[layout_name]["BC_test+BC_test"] = mean_and_std_err(eval_trajs['ep_returns'])
        pickle.dump(eval_trajs, open('saved_eval_trajs/bc_test_and_bc_test_ex4_dual.pkl', 'wb'))

        # Evaluate the BC test model playing with BC train. Get mean and stderr of the returns of rollouts. BC train
        # is Player 0. This is "BC_train+BC_test_0".
        bc_train, bc_params_train = get_bc_agent_from_saved(bc_model_paths['train'][layout_name])
        bc_test, bc_params_test = get_bc_agent_from_saved(bc_model_paths['test'][layout_name])
        del bc_params_train["data_params"]
        del bc_params_test["data_params"]
        assert common_keys_equal(bc_params_train, bc_params_test)
        ae = AgentEvaluator(mdp_params=bc_params_train["mdp_params"], env_params=bc_params_train["env_params"])

        train_and_test = ae.evaluate_agent_pair(AgentPair(bc_train, bc_test), num_games=num_rounds)
        best_bc_models_performance[layout_name]["BC_train+BC_test_0"] = mean_and_std_err(train_and_test['ep_returns'])
        pickle.dump(eval_trajs, open('saved_eval_trajs/bc_train_and_bc_test_0_ex4_dual.pkl', 'wb'))

        # Evaluate the BC test model playing with BC train. Get mean and stderr of the returns of rollouts. Swap the order. BC train
        # is Player 1. This is "BC_train+BC_test_1".
        test_and_train = ae.evaluate_agent_pair(AgentPair(bc_test, bc_train), num_games=num_rounds)
        best_bc_models_performance[layout_name]["BC_train+BC_test_1"] = mean_and_std_err(test_and_train['ep_returns'])
        pickle.dump(eval_trajs, open('saved_eval_trajs/bc_train_and_bc_test_1_ex4_dual.pkl', 'wb'))

    return best_bc_models_performance


def run_fixed_bc_experiments():
    # Train BC models
    # seeds = [5415, 2652, 6440, 1965, 6647]
    seed = 5415
    # num_seeds = len(seeds)

    # params_unident = {"layout_name": "unident_s", "num_epochs": 120, "lr": 1e-3, "adam_eps": 1e-8}
    # params_simple = {"layout_name": "simple", "num_epochs": 100, "lr": 1e-3, "adam_eps":1e-8}
    # params_random1 = {"layout_name": "random1", "num_epochs": 120, "lr": 1e-3, "adam_eps":1e-8}

    params_random0_default = {"layout_name": "random0", "num_epochs": 90, "lr": 1e-3, "adam_eps":1e-8}
    params_random0 = {"layout_name": "random0", "num_epochs": 180, "lr": 1e-4, "adam_eps": 1e-8}

    # params_random3 = {"layout_name": "random3", "num_epochs": 110, "lr": 1e-3, "adam_eps":1e-5}

    # all_params = [params_simple, params_random1, params_unident, params_random0, params_random3]
    # all_params = [params_unident]

    # 1. Train BC models for all parameters and seeds
    # train_bc_models(all_params, seeds)
    train_bc_models_on_fixed_agent_trajs(a0_type='SP', a1_type='SP', N_teams=12, params=params_random0, seed=seed)

    # Evaluate BC models
    # rollout_two_bc_agents(a0_path=None, a1_path=None)


def rollout_two_bc_agents(a0_path=None, a1_path=None):
    reset_tf()

    # seeds = {
    #     "bc_train": [9456, 1887, 5578, 5987,  516],
    #     "bc_test": [2888, 7424, 7360, 4467,  184]
    # }
    seeds = {
        "bc_train": [9456, 1887, 5578, 5987, 516],
        "bc_test": [9456, 1887, 5578, 5987, 516],
    }

    selected_bc_model_paths = {
        'train': {
            # "random0": "dual_pot_finetune_random0_bc_train_seed5415",
            # "random0": "dual_pot_finetune_random0_bc_train_seed5415",
            "random0": "single_pot_finetune_random0_bc_train_seed5415",
            # "random0": "dual_pot_finetune_random0_bc_test_seed5415",
            # "random0": "single_pot_finetune_random0_bc_test_seed5415",
            # "random0": "random0_bc_train_seed0",
            # "random0": "orig_berk_random0_bc_test_seed2",
        },
        'test': {
            # "random0": "dual_pot_finetune_random0_bc_train_seed5415",
            # "random0": "dual_pot_finetune_random0_bc_train_seed5415",
            "random0": "single_pot_finetune_random0_bc_train_seed5415",
            # "random0": "dual_pot_finetune_random0_bc_test_seed5415",
            # "random0": "single_pot_finetune_random0_bc_test_seed5415",
            # "random0": "random0_bc_train_seed0", # original train
            # "random0": "orig_berk_random0_bc_test_seed2", # bc test
        }
    }
    if a0_path is not None:
        selected_bc_model_paths['train'] = a0_path
    if a1_path is not None:
        selected_bc_model_paths['test'] = a1_path


    set_global_seed(248)
    num_rounds = 100
    bc_bc_performance = evaluate_two_bc_models(best_bc_model_paths, num_rounds, seeds, best=True)
    print('BC vs. BC bc_bc_performance', bc_bc_performance)
    # bc_bc_performance  = prepare_nested_default_dict_for_pickle(bc_bc_performance)
    # save_pickle(ppo_bc_performance, PPO_DATA_DIR + "ppo_bc_models_performance")

def evaluate_two_bc_models(best_bc_model_paths, num_rounds, seeds, best):
    # evaluate_all_ppo_bc_models(ppo_bc_model_paths, best_bc_model_paths, num_rounds, seeds, best=True)
    layouts = list(best_bc_model_paths['train'].keys())
    ppo_bc_performance = {}
    for layout in layouts:
        print(layout)
        layout_eval = run_two_bc_models_for_layout(layout, num_rounds, best_bc_model_paths, seeds=seeds, best=best)
        ppo_bc_performance.update(dict(layout_eval))
    return ppo_bc_performance

def run_two_bc_models_for_layout(layout, num_rounds, bc_model_paths, seeds, best=False,
                                          display=False):
    # evaluate_ppo_and_bc_models_for_layout(layout, num_rounds, best_bc_model_paths, ppo_bc_model_paths, seeds=seeds, best=best)
    # assert len(seeds["bc_train"]) == len(seeds["bc_test"])

    bc_bc_performance = defaultdict(lambda: defaultdict(list))

    agent_bc_train, train_bc_params = get_bc_agent_from_saved(bc_model_paths['train'][layout])
    agent_bc_test, test_bc_params = get_bc_agent_from_saved(bc_model_paths['test'][layout])

    # print('train_bc_params', train_bc_params)

    evaluator = AgentEvaluator(mdp_params=train_bc_params["mdp_params"], env_params=train_bc_params["env_params"])


    # assert common_keys_equal(bc_params["mdp_params"], ppo_config["mdp_params"])

    # Play two agents against each other.
    bc_and_bc = evaluator.evaluate_agent_pair(AgentPair(agent_bc_train, agent_bc_test, allow_duplicate_agents=False), num_games=max(int(num_rounds), 1), display=display)
    avg_bc_and_bc = np.mean(bc_and_bc['ep_returns'])
    bc_bc_performance[layout]["BC_train+BC_test"].append(avg_bc_and_bc)

    # Swap order and Play two agents against each other.
    bc_and_bc = evaluator.evaluate_agent_pair(
        AgentPair(agent_bc_test, agent_bc_train, allow_duplicate_agents=False),
        num_games=max(int(num_rounds), 1), display=display)
    avg_bc_and_bc = np.mean(bc_and_bc['ep_returns'])
    bc_bc_performance[layout]["BC_test+BC_train"].append(avg_bc_and_bc)


    return bc_bc_performance



if __name__ == "__main__":
    run_fixed_bc_experiments()

