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
from human_aware_rl.imitation.behavioural_cloning import train_bc_agent, eval_with_benchmarking_from_saved,BC_SAVE_DIR, plot_bc_run, \
    DEFAULT_BC_PARAMS, get_bc_agent_from_saved, plot_bc_run_modified, train_bc_agent_w_finetuning


# Path for dict containing the best bc models paths
BEST_BC_MODELS_PATH = BC_SAVE_DIR + "best_bc_model_paths"
BC_MODELS_EVALUATION_PATH = BC_SAVE_DIR + "bc_models_all_evaluations"

def train_bc_agent_from_hh_data(selected_workers, save_filename, layout_name, agent_name, num_epochs, lr, adam_eps, model, finetuning=False):
    """Trains a BC agent from human human data (model can either be `train` or `test`, which is trained
    on two different subsets of the data)."""

    bc_params = copy.deepcopy(DEFAULT_BC_PARAMS)
    bc_params["data_params"]['train_mdps'] = [layout_name]
    bc_params["data_params"]['data_path'] = "../data/human/anonymized/clean_{}_trials.pkl".format(model)
    bc_params["mdp_params"]['layout_name'] = layout_name
    bc_params["mdp_params"]['start_order_list'] = None

    # model_save_dir = "aa_strat3_finetune_" + layout_name + "_" + agent_name + "/"
    model_save_dir = save_filename
    # model_save_dir = layout_name + "_" + agent_name + "/"
    # train_bc_agent() returns the bc model
    # if model == 'train':
    #     is_train = True
    # else:
    #     is_train = False

    if finetuning:
        return train_bc_agent_w_finetuning(selected_workers, model_save_dir, bc_params, num_epochs=num_epochs, lr=lr, adam_eps=adam_eps)
    return train_bc_agent(model_save_dir, bc_params, num_epochs=num_epochs, lr=lr, adam_eps=adam_eps)

def train_bc_models(all_params, seeds):
    """Train len(seeds) num of models for each layout"""
    for params in all_params:
        for seed_idx, seed in enumerate(seeds):
            set_global_seed(seed)
            # 1. Train BC model
            model = train_bc_agent_from_hh_data(agent_name="bc_train_seed{}".format(seed_idx), model='train', **params)
            # plot_bc_run(model.bc_info, params['num_epochs'])
            plot_bc_run_modified(model.bc_info, params['num_epochs'], seed_idx, seed)

            # 2. Test BC model
            model = train_bc_agent_from_hh_data(agent_name="bc_test_seed{}".format(seed_idx), model='test', **params)
            # plot_bc_run(model.bc_info, params['num_epochs'])
            plot_bc_run_modified(model.bc_info, params['num_epochs'], seed_idx, seed)
            reset_tf()

def get_strategy_dicts():
    layout_strategy_dict = {
        "unident_s": {
            0: [6, 11, 16, 31, 51, 56, 61, 66, 71, 76, 86, 91, 96, 101, 111, 116],
            1: [21, 46, 81],
        },
        "simple": {
            0: [70, 75, 80],
            1: [5, 10, 45, 60, 100],
            2: [15, 20, 50, 55, 65, 85, 90, 110, 115],
            3: [0, 95],
        },
        "random0": {
            0: [19, 24, 69, 79, 89, 114],
            1: [9, 14, 54, 59, 64, 99],
        },
        "random1": {
            0: [7, 12, 17, 22, 62, 67, 72, 82, 87, 92, 97, 102, 112],
            1: [47, 52, 117],
        },
        "random3": {
            0: [63, 73],
            1: [48, 53, 83, 93],
            2: [8, 18, 33, 58, 68, 88, 98, 103, 113],
            3: [13, 23, 78],
        },

    }

    layout_id_to_worker_num = {
        "unident_s": {0: None, 1: 6, 2: 11, 3: 16, 4: 21, 6: 31, 9: 46, 10: 51, 11: 56, 12: 61, 13: 66, 14: 71, 15: 76,
                      16: 81, 17: 86, 18: 91, 19: 96, 20: 101, 22: 111, 23: 116},
        "simple": {0: 0, 1: 5, 2: 10, 3: 15, 4: 20, 6: None, 9: 45, 10: 50, 11: 55, 12: 60, 13: 65, 14: 70, 15: 75,
                   16: 80, 17: 85, 18: 90, 19: 95, 20: 100, 22: 110, 23: 115},
        "random0": {0: None, 1: 9, 2: 14, 3: 19, 4: 24, 6: None, 9: None, 10: 54, 11: 59, 12: 64, 13: 69, 14: None,
                    15: 79, 16: None, 17: 89, 18: None, 19: 99, 20: None, 22: 114, 23: None},
        "random1": {0: None, 1: 7, 2: 12, 3: 17, 4: 22, 6: None, 9: 47, 10: 52, 11: 57, 12: 62, 13: 67, 14: 72, 15: 77,
                    16: 82, 17: 87, 18: 92, 19: 97, 20: 102, 22: 112, 23: 117},
        "random3": {0: None, 1: 8, 2: 13, 3: 18, 4: 23, 6: 33, 9: 48, 10: 53, 11: 58, 12: 63, 13: 68, 14: 73, 15: 78,
                    16: 83, 17: 88, 18: 93, 19: 98, 20: 103, 22: 113, 23: None},
    }
    worker_num_to_layout_id = {}
    for layout_to_run in layout_id_to_worker_num:
        worker_num_to_layout_id[layout_to_run] = {v: k for k, v in layout_id_to_worker_num[layout_to_run].items()}

    for layout_to_run in layout_strategy_dict:
        for strat_idx in layout_strategy_dict[layout_to_run]:
            layout_strategy_dict[layout_to_run][strat_idx] = [worker_num_to_layout_id[layout_to_run][elem] for elem in
                                                              layout_strategy_dict[layout_to_run][strat_idx]]
            # worker_num_list = []
            # for elem in layout_strategy_dict[layout_to_run][strat_idx]:
            #     worker_num_list.append(worker_num_to_layout_id[layout_to_run][elem])
    return layout_strategy_dict, layout_id_to_worker_num, worker_num_to_layout_id

def train_bc_models_w_finetuning_on_single_strat_train(all_params, seeds):
    """Train len(seeds) num of models for each layout"""
    layout_strategy_dict, layout_id_to_worker_num, worker_num_to_layout_id = get_strategy_dicts()
    for params in all_params:
        layout_name = params["layout_name"]
        all_strategies = list(layout_strategy_dict[layout_name].keys())
        for strategy_idx in all_strategies:
            selected_workers = layout_strategy_dict[layout_name][strategy_idx]
            for seed_idx, seed in enumerate(seeds):
                set_global_seed(seed)
                # 1. Train BC model


                save_filename = layout_name+"_STRAT"+str(strategy_idx)+"_finetune3070_" + "seed"+ str(seed) + "/"

                model = train_bc_agent_from_hh_data(selected_workers, save_filename, agent_name="bc_train_seed{}".format(seed), model='train', **params, finetuning=True)
                # plot_bc_run(model.bc_info, params['num_epochs'])
                plot_bc_run_modified(model.bc_info, params['num_epochs'], seed_idx, seed)

                # 2. Test BC model
                # model = train_bc_agent_from_hh_data(agent_name="bc_test_seed{}".format(seed), model='test', **params, finetuning=False)
                # # plot_bc_run(model.bc_info, params['num_epochs'])
                # plot_bc_run_modified(model.bc_info, params['num_epochs'], seed_idx, seed)
                reset_tf()

def evaluate_strategy_bc_models(all_params, num_rounds, seeds):
    """Evaluate all trained models"""
    layout_strategy_dict, layout_id_to_worker_num, worker_num_to_layout_id = get_strategy_dicts()

    bc_models_evaluation = {}
    for params in all_params:
        layout_name = params["layout_name"]
        all_strategies = list(layout_strategy_dict[layout_name].keys())
        
        print(layout_name)
        bc_models_evaluation[layout_name] = { "train": {}, "test": {} }

        for strategy_idx in all_strategies:
            selected_workers = layout_strategy_dict[layout_name][strategy_idx]
            for seed_idx, seed in enumerate(seeds):
                set_global_seed(seed)
                # 1. Train BC model

                save_filename = layout_name + "_strat" + str(strategy_idx) + "_finetune_" + "seed" + str(seed) + "/"

            # For all params and seeds, evaluate the model with a saved model. Pass in the BC model file name to evaluate.
                eval_trajs = eval_with_benchmarking_from_saved(num_rounds, save_filename)
                bc_models_evaluation[layout_name]["train"][strategy_idx] = np.mean(eval_trajs['ep_returns'])
                # pickle.dump(eval_trajs, open(f'saved_eval_trajs/{save_filename}.pkl', 'wb'))
                #
                # eval_trajs = eval_with_benchmarking_from_saved(num_rounds, layout_name + "_bc_test_seed{}".format(seed_idx))
                # bc_models_evaluation[layout_name]["test"][seed_idx] = np.mean(eval_trajs['ep_returns'])
                # pickle.dump(eval_trajs, open('saved_eval_trajs/test_test_ex4_dual.pkl', 'wb'))

    return bc_models_evaluation


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

def run_all_bc_experiments():
    # Train BC models
    # seeds = [5415, 2652, 6440, 1965, 6647]
    seeds = [5415]
    num_seeds = len(seeds)

    params_unident = {"layout_name": "unident_s", "num_epochs": 120*2, "lr": 1e-3, "adam_eps":1e-8}
    params_simple = {"layout_name": "simple", "num_epochs": 100*2, "lr": 1e-3, "adam_eps":1e-8}
    params_random1 = {"layout_name": "random1", "num_epochs": 120*2, "lr": 1e-3, "adam_eps":1e-8}
    params_random0 = {"layout_name": "random0", "num_epochs": 90*2, "lr": 1e-3, "adam_eps":1e-8}
    params_random3 = {"layout_name": "random3", "num_epochs": 110*2, "lr": 1e-3, "adam_eps":1e-8}

    all_params = [params_simple, params_random1, params_unident, params_random0, params_random3]
    # all_params = [params_unident]

    # 1. Train BC models for all parameters and seeds
    # train_bc_models(all_params, seeds)
    train_bc_models_w_finetuning_on_single_strat_train(all_params, seeds)

    # Evaluate BC models
    set_global_seed(64)

    num_rounds = 5
    bc_models_evaluation = evaluate_all_bc_models(all_params, num_rounds, num_seeds)

    # Save evaluation to BC_MODELS_EVALUATION_PATH
    print('BC_MODELS_EVALUATION_PATH = ', BC_MODELS_EVALUATION_PATH)
    save_pickle(bc_models_evaluation, BC_MODELS_EVALUATION_PATH)
    print("All BC models evaluation: ", bc_models_evaluation)

    # These models have been manually selected to more or less match in performance,
    # (test BC model should be a bit better than the train BC model)
    selected_models = {
        # "simple": [0, 1],
        # "unident_s": [0, 0],
        # "random1": [4, 2],
        "random0": [2, 1],
        # "random3": [3, 3]
    }
    # selected_models = {
    #     "random0": [0, 0],
    # }
    # selected_models = {
    #     "unident_s": [0, 0],
    # }

    final_bc_model_paths = { "train": {}, "test": {} }
    for layout_name, seed_indices in selected_models.items():
        train_idx, test_idx = seed_indices
        # Index 0 is the train model
        # Index 1 is the test model
        # Record the layout model path names
        final_bc_model_paths["train"][layout_name] = "{}_bc_train_seed{}".format(layout_name, train_idx)
        final_bc_model_paths["test"][layout_name] = "{}_bc_test_seed{}".format(layout_name, test_idx)
    #
    # # Evaluate the BC models defined by selected models
    best_bc_models_performance = evaluate_bc_models(final_bc_model_paths, num_rounds)
    print('best_bc_models_performance = ', best_bc_models_performance)
    # save_pickle(best_bc_models_performance, BC_SAVE_DIR + "best_bc_models_performance")
    

if __name__ == "__main__":
    # with open("../data/bc_runs/true_berk_bc_models_all_evaluations.pickle", "rb") as file:
    #     bc_eval = pickle.load(file)
    #
    # # bc_eval = {'simple': {'train': {0: 102.0, 1: 96.0, 2: 110.0, 3: 118.0}, 'test': {}}, 'random1': {'train': {0: 84.0, 1: 76.0}, 'test': {}}, 'unident_s': {'train': {0: 170.0, 1: 86.0}, 'test': {}}, 'random0': {'train': {0: 28.0, 1: 40.0}, 'test': {}}, 'random3': {'train': {0: 40.0, 1: 58.0, 2: 54.0, 3: 58.0}, 'test': {}}}
    # # print("bc_eval", bc_eval)
    # avg_results = {}
    # for layout_name in bc_eval:
    #     all_vals = []
    #     for trial_name in ['train']:
    #         all_vals.extend(list(bc_eval[layout_name][trial_name].values()))
    #         # print("bc_eval[layout_name][trial_name]", bc_eval[layout_name][trial_name])
    #     avg_results[layout_name] = np.mean(list(bc_eval[layout_name][trial_name].values()))
    #
    # print("avg_results", avg_results)

    seeds = [5415]
    num_seeds = len(seeds)
    num_rounds = 10

    params_unident = {"layout_name": "unident_s", "num_epochs": 120 * 2, "lr": 1e-3, "adam_eps": 1e-8}
    params_simple = {"layout_name": "simple", "num_epochs": 100 * 2, "lr": 1e-3, "adam_eps": 1e-8}
    params_random1 = {"layout_name": "random1", "num_epochs": 120 * 2, "lr": 1e-3, "adam_eps": 1e-8}
    params_random0 = {"layout_name": "random0", "num_epochs": 90 * 2, "lr": 1e-3, "adam_eps": 1e-8}
    params_random3 = {"layout_name": "random3", "num_epochs": 110 * 2, "lr": 1e-3, "adam_eps": 1e-8}

    all_params = [params_simple, params_random1, params_unident, params_random0, params_random3]

    bc_models_evaluation = evaluate_strategy_bc_models(all_params, num_rounds, seeds)

    # Save evaluation to BC_MODELS_EVALUATION_PATH
    print('BC_MODELS_EVALUATION_PATH = ', BC_MODELS_EVALUATION_PATH)
    # save_pickle(bc_models_evaluation, BC_MODELS_EVALUATION_PATH)
    print("All BC models evaluation: ", bc_models_evaluation)
    # run_all_bc_experiments()

    bc_eval = bc_models_evaluation
    avg_results = {}
    for layout_name in bc_eval:
        all_vals = []
        for trial_name in ['train']:
            all_vals.extend(list(bc_eval[layout_name][trial_name].values()))
            # print("bc_eval[layout_name][trial_name]", bc_eval[layout_name][trial_name])
        avg_results[layout_name] = np.mean(list(bc_eval[layout_name][trial_name].values()))

    print("avg_results", avg_results)


# Automatic selection of best BC models. Caused imbalances that made interpretation of results more difficult, 
# better to select manually non-best ones.

# def select_bc_models(bc_models_evaluation, num_rounds, num_seeds):
#     best_bc_model_paths = { "train": {}, "test": {} }

#     for layout_name, layout_eval_dict in bc_models_evaluation.items():
#         for model_type, seed_eval_dict in layout_eval_dict.items():
#             best_seed = np.argmax([seed_eval_dict[i] for i in range(num_seeds)])
#             best_bc_model_paths[model_type][layout_name] = "{}_bc_{}_seed{}".format(layout_name, model_type, best_seed)

#     save_pickle(best_bc_model_paths, BEST_BC_MODELS_PATH)
#     return best_bc_model_paths