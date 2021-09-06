import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import defaultdict
import pickle

from overcooked_ai_py.utils import save_pickle, load_pickle
from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.utils import save_pickle

from human_aware_rl.utils import reset_tf, set_global_seed, prepare_nested_default_dict_for_pickle, common_keys_equal
from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved
from human_aware_rl.ppo.ppo import get_ppo_agent, plot_ppo_run, PPO_DATA_DIR


def plot_runs_training_curves(ppo_bc_model_paths, seeds, single=False, show=False, save=False):
    # Plot PPO BC models
    for run_type, type_dict in ppo_bc_model_paths.items():
        print(run_type)
        if 'test' in run_type:
            continue
        for layout, layout_model_path in type_dict.items():
            print(layout)
            plt.figure(figsize=(8, 5))
            plot_ppo_run(layout_model_path, sparse=True, print_config=False, single=single, seeds=seeds[run_type])
            plt.xlabel("Environment timesteps")
            plt.ylabel("Mean episode reward")
            if save: plt.savefig("strat_specific_singlewft_rew_ppo_bc_{}_{}".format(run_type, layout),
                                 bbox_inches='tight')
            if show: plt.show()


def evaluate_ppo_and_bc_models_for_layout(layout, num_rounds, bc_model_paths, ppo_bc_model_paths, seeds, best=False,
                                          display=False):
    # evaluate_ppo_and_bc_models_for_layout(layout, num_rounds, best_bc_model_paths, ppo_bc_model_paths, seeds=seeds, best=best)
    assert len(seeds["bc_train"]) == len(seeds["bc_test"])
    ppo_bc_performance = defaultdict(lambda: defaultdict(list))

    agent_bc_test, bc_params = get_bc_agent_from_saved(bc_model_paths['test'][layout])
    ppo_bc_train_path = ppo_bc_model_paths['bc_train'][layout]
    ppo_bc_test_path = ppo_bc_model_paths['bc_test'][layout]
    evaluator = AgentEvaluator(mdp_params=bc_params["mdp_params"], env_params=bc_params["env_params"])

    for seed_idx in range(len(seeds["bc_train"])):
        agent_ppo_bc_train, ppo_config = get_ppo_agent(ppo_bc_train_path, seeds["bc_train"][seed_idx], best=best)
        assert common_keys_equal(bc_params["mdp_params"], ppo_config["mdp_params"])

        # For curiosity, how well does agent do with itself?
        # ppo_and_ppo = evaluator.evaluate_agent_pair(AgentPair(agent_ppo_bc_train, agent_ppo_bc_train, allow_duplicate_agents=True), num_games=max(int(num_rounds/2), 1), display=display)
        # avg_ppo_and_ppo = np.mean(ppo_and_ppo['ep_returns'])
        # ppo_bc_performance[layout]["PPO_BC_train+PPO_BC_train"].append(avg_ppo_and_ppo)

        # How well it generalizes to new agent in simulation?
        ppo_and_bc = evaluator.evaluate_agent_pair(AgentPair(agent_ppo_bc_train, agent_bc_test), num_games=num_rounds,
                                                   display=display)
        avg_ppo_and_bc = np.mean(ppo_and_bc['ep_returns'])
        ppo_bc_performance[layout]["PPO_BC_train+BC_test_0"].append(avg_ppo_and_bc)

        bc_and_ppo = evaluator.evaluate_agent_pair(AgentPair(agent_bc_test, agent_ppo_bc_train), num_games=num_rounds,
                                                   display=display)
        avg_bc_and_ppo = np.mean(bc_and_ppo['ep_returns'])
        ppo_bc_performance[layout]["PPO_BC_train+BC_test_1"].append(avg_bc_and_ppo)

        # How well could we do if we knew true model BC_test?
        agent_ppo_bc_test, ppo_config = get_ppo_agent(ppo_bc_test_path, seeds["bc_test"][seed_idx], best=best)
        assert common_keys_equal(bc_params["mdp_params"], ppo_config["mdp_params"])

        ppo_and_bc = evaluator.evaluate_agent_pair(AgentPair(agent_ppo_bc_test, agent_bc_test), num_games=num_rounds,
                                                   display=display)
        avg_ppo_and_bc = np.mean(ppo_and_bc['ep_returns'])
        ppo_bc_performance[layout]["PPO_BC_test+BC_test_0"].append(avg_ppo_and_bc)

        bc_and_ppo = evaluator.evaluate_agent_pair(AgentPair(agent_bc_test, agent_ppo_bc_test), num_games=num_rounds,
                                                   display=display)
        avg_bc_and_ppo = np.mean(bc_and_ppo['ep_returns'])
        ppo_bc_performance[layout]["PPO_BC_test+BC_test_1"].append(avg_bc_and_ppo)

    return ppo_bc_performance


def run_two_ppo_models_for_layout(layout, num_rounds, ppo_bc_model_paths, bc_model_paths, seeds, best=False,
                                  display=False):
    # evaluate_ppo_and_bc_models_for_layout(layout, num_rounds, best_bc_model_paths, ppo_bc_model_paths, seeds=seeds, best=best)
    # assert len(seeds["bc_train"]) == len(seeds["bc_test"])
    ppo_bc_performance = defaultdict(lambda: defaultdict(list))

    agent_bc_test, bc_params = get_bc_agent_from_saved(bc_model_paths['test'][layout])
    ppo_bc_train_path = ppo_bc_model_paths['bc_train'][layout]
    ppo_bc_test_path = ppo_bc_model_paths['bc_test'][layout]

    evaluator = AgentEvaluator(mdp_params=bc_params["mdp_params"], env_params=bc_params["env_params"])

    for train_seed_idx in range(len(seeds["bc_train"])):
        for test_seed_idx in range(len(seeds["bc_test"])):
            agent_ppo_bc_train, ppo_config = get_ppo_agent(ppo_bc_train_path, seeds["bc_train"][train_seed_idx],
                                                           best=best)
            agent_ppo_bc_test, ppo_test_config = get_ppo_agent(ppo_bc_test_path, seeds["bc_test"][test_seed_idx],
                                                               best=best)

            # assert common_keys_equal(bc_params["mdp_params"], ppo_config["mdp_params"])

            # Play two agents against each other.
            ppo_and_ppo = evaluator.evaluate_agent_pair(
                AgentPair(agent_ppo_bc_train, agent_ppo_bc_test, allow_duplicate_agents=False),
                num_games=max(int(num_rounds / 2), 1), display=display)
            avg_ppo_and_ppo = np.mean(ppo_and_ppo['ep_returns'])
            ppo_bc_performance[layout]["PPO_BC_train+PPO_BC_test"].append(avg_ppo_and_ppo)

            # Swap order and Play two agents against each other.
            ppo_and_ppo = evaluator.evaluate_agent_pair(
                AgentPair(agent_ppo_bc_test, agent_ppo_bc_train, allow_duplicate_agents=False),
                num_games=max(int(num_rounds / 2), 1), display=display)
            avg_ppo_and_ppo = np.mean(ppo_and_ppo['ep_returns'])
            ppo_bc_performance[layout]["PPO_BC_test+PPO_BC_train"].append(avg_ppo_and_ppo)

    return ppo_bc_performance


def run_two_bc_models_for_layout(layout, num_rounds, bc_model_paths, seeds, best=False,
                                 display=False):
    # evaluate_ppo_and_bc_models_for_layout(layout, num_rounds, best_bc_model_paths, ppo_bc_model_paths, seeds=seeds, best=best)
    # assert len(seeds["bc_train"]) == len(seeds["bc_test"])

    bc_bc_performance = defaultdict(lambda: defaultdict(list))

    agent_bc_train, train_bc_params = get_bc_agent_from_saved(bc_model_paths['train'][layout])
    agent_bc_test, test_bc_params = get_bc_agent_from_saved(bc_model_paths['test'][layout])

    evaluator = AgentEvaluator(mdp_params=train_bc_params["mdp_params"], env_params=train_bc_params["env_params"])

    # assert common_keys_equal(bc_params["mdp_params"], ppo_config["mdp_params"])

    # Play two agents against each other.
    bc_and_bc = evaluator.evaluate_agent_pair(AgentPair(agent_bc_train, agent_bc_test, allow_duplicate_agents=False),
                                              num_games=max(int(num_rounds), 1), display=display)
    avg_bc_and_bc = np.mean(bc_and_bc['ep_returns'])
    bc_bc_performance[layout]["BC_train+BC_test"].append(avg_bc_and_bc)

    # Swap order and Play two agents against each other.
    bc_and_bc = evaluator.evaluate_agent_pair(
        AgentPair(agent_bc_test, agent_bc_train, allow_duplicate_agents=False),
        num_games=max(int(num_rounds), 1), display=display)
    avg_bc_and_bc = np.mean(bc_and_bc['ep_returns'])
    bc_bc_performance[layout]["BC_test+BC_train"].append(avg_bc_and_bc)

    return bc_bc_performance


def evaluate_two_ppo_models(ppo_bc_model_paths, best_bc_model_paths, num_rounds, seeds, best):
    # evaluate_all_ppo_bc_models(ppo_bc_model_paths, best_bc_model_paths, num_rounds, seeds, best=True)
    layouts = list(ppo_bc_model_paths['bc_train'].keys())
    ppo_bc_performance = {}
    for layout in layouts:
        print(layout)
        layout_eval = run_two_ppo_models_for_layout(layout, num_rounds, ppo_bc_model_paths, best_bc_model_paths,
                                                    seeds=seeds, best=best)
        ppo_bc_performance.update(dict(layout_eval))
    return ppo_bc_performance


def evaluate_two_bc_models(best_bc_model_paths, num_rounds, seeds, best):
    # evaluate_all_ppo_bc_models(ppo_bc_model_paths, best_bc_model_paths, num_rounds, seeds, best=True)
    layouts = list(best_bc_model_paths['train'].keys())
    ppo_bc_performance = {}
    for layout in layouts:
        print(layout)
        layout_eval = run_two_bc_models_for_layout(layout, num_rounds, best_bc_model_paths, seeds=seeds, best=best)
        ppo_bc_performance.update(dict(layout_eval))
    return ppo_bc_performance


def evaluate_all_ppo_bc_models(ppo_bc_model_paths, best_bc_model_paths, num_rounds, seeds, best):
    # evaluate_all_ppo_bc_models(ppo_bc_model_paths, best_bc_model_paths, num_rounds, seeds, best=True)
    layouts = list(ppo_bc_model_paths['bc_train'].keys())
    ppo_bc_performance = {}
    for layout in layouts:
        print(layout)
        layout_eval = evaluate_ppo_and_bc_models_for_layout(layout, num_rounds, best_bc_model_paths, ppo_bc_model_paths,
                                                            seeds=seeds, best=best)
        ppo_bc_performance.update(dict(layout_eval))
    return ppo_bc_performance


def run_all_ppo_bc_experiments(best_bc_model_paths):
    reset_tf()

    seeds = {
        "bc_train": [9456, 1887, 5578, 5987, 516],
        "bc_test": [2888, 7424, 7360, 4467, 184]
    }

    ppo_bc_model_paths = {
        'bc_train': {
            "simple": "ppo_bc_train_simple",
            "unident_s": "ppo_bc_train_unident_s",
            "random1": "ppo_bc_train_random1",
            "random0": "ppo_bc_train_random0",
            "random3": "ppo_bc_train_random3"
        },
        'bc_test': {
            "simple": "ppo_bc_test_simple",
            "unident_s": "ppo_bc_test_unident_s",
            "random1": "ppo_bc_test_random1",
            "random0": "ppo_bc_test_random0",
            "random3": "ppo_bc_test_random3"
        }
    }

    plot_runs_training_curves(ppo_bc_model_paths, seeds, save=True)

    set_global_seed(248)
    num_rounds = 100
    ppo_bc_performance = evaluate_all_ppo_bc_models(ppo_bc_model_paths, best_bc_model_paths, num_rounds, seeds,
                                                    best=True)
    ppo_bc_performance = prepare_nested_default_dict_for_pickle(ppo_bc_performance)
    save_pickle(ppo_bc_performance, PPO_DATA_DIR + "ppo_bc_models_performance")


#########################################################################################################
# My PPO Modifications
#########################################################################################################
def check_replicate_evaluate_all_ppo_bc_experiments(best_bc_model_paths):
    reset_tf()

    seeds = {
        "bc_train": [9456, 1887, 5578, 5987, 516],
        "bc_test": [2888, 7424, 7360, 4467, 184]
    }

    best_bc_model_paths = {
        'train': {
            "random0": "random0_bc_train_seed0",
        },
        'test': {
            "random0": "orig_berk_random0_bc_test_seed2",
        }
    }
    # best_bc_model_paths = {
    #     'train': {
    #         "random0": "dual_pot_finetune_random0_bc_train_seed5415",
    #     },
    #     'test': {
    #         "random0": "dual_pot_finetune_random0_bc_train_seed5415",
    #     }
    # }
    # best_bc_model_paths = {
    #     'train': {
    #         "random0": "single_pot_finetune_random0_bc_train_seed5415",
    #     },
    #     'test': {
    #         "random0": "single_pot_finetune_random0_bc_train_seed5415",
    #     }
    # }

    ppo_bc_model_paths = {
        'bc_train': {
            # "random0": "2021_07_28-16_53_15_single_strat_wft_ppo_bc_train_random0_test4",
            # "random0": "2021_07_28-16_50_14_dual_strat_nft_ppo_bc_train_random0_test3",
            "random0": "../../../data/ppo_runs/2021_08_19-13_36_51_dual_strat_wft_ppo_bc_train_random0_test5",
        },
        'bc_test': {
            # "random0": "orig_berk_random0_bc_test_seed2",
            # "random0": "2021_07_30-00_59_56_single_strat_wft_ppo_bc_test_random0_test4",
            # "random0": "2021_07_30-01_20_34_dual_strat_nft_ppo_bc_test_random0_test3",
            "random0": "../../../data/ppo_runs/2021_08_19-23_18_20_dual_strat_wft_ppo_bc_test_random0_test5",
        }
    }

    plot_runs_training_curves(ppo_bc_model_paths, seeds, save=True)

    set_global_seed(248)
    num_rounds = 100
    ppo_bc_performance = evaluate_all_ppo_bc_models(ppo_bc_model_paths, best_bc_model_paths, num_rounds, seeds,
                                                    best=True)
    print('SINGLE ppo_bc_performance', ppo_bc_performance)
    ppo_bc_performance = prepare_nested_default_dict_for_pickle(ppo_bc_performance)
    # save_pickle(ppo_bc_performance, PPO_DATA_DIR + "ppo_bc_models_performance")


def run_two_ppo_agents():
    reset_tf()

    # seeds = {
    #     "bc_train": [9456, 1887, 5578, 5987,  516],
    #     "bc_test": [2888, 7424, 7360, 4467,  184]
    # }
    seeds = {
        "bc_train": [9456, 1887, 5578, 5987, 516],
        "bc_test": [9456, 1887, 5578, 5987, 516],
    }
    # seeds = {
    #     "bc_train": [2888, 7424, 7360, 4467,  184],
    #     "bc_test": [2888, 7424, 7360, 4467,  184],
    # }

    best_bc_model_paths = {
        'train': {
            "random0": "dual_pot_finetune_random0_bc_train_seed5415",
        },
        'test': {
            "random0": "dual_pot_finetune_random0_bc_train_seed5415",
            # "random0": "dual_pot_finetune_random0_bc_test_seed5415",
        }
    }

    ppo_bc_model_paths = {
        'bc_train': {
            # "random0": "2021_07_28-16_53_15_single_strat_wft_ppo_bc_train_random0_test4",
            # "random0": "2021_07_28-16_50_14_dual_strat_nft_ppo_bc_train_random0_test3",
            # "random0": "2021_07_30-01_20_34_dual_strat_nft_ppo_bc_test_random0_test3",
            "random0": "../../../data/ppo_runs/2021_08_19-13_36_51_dual_strat_wft_ppo_bc_train_random0_test5",
        },
        'bc_test': {
            # "random0": "orig_berk_random0_bc_test_seed2",
            "random0": "2021_07_28-16_53_15_single_strat_wft_ppo_bc_train_random0_test4",
            # "random0": "2021_07_28-16_53_15_single_strat_wft_ppo_bc_train_random0_test4",
            # "random0": "2021_07_30-01_20_34_dual_strat_nft_ppo_bc_test_random0_test3",
            # "random0": "2021_07_28-16_50_14_dual_strat_nft_ppo_bc_train_random0_test3",
            # "random0": "2021_07_30-00_59_56_single_strat_wft_ppo_bc_test_random0_test4",
            # "random0": "../../../data/ppo_runs/2021_08_19-13_36_51_dual_strat_wft_ppo_bc_train_random0_test5",
            # "random0": "../../../data/ppo_runs/2021_08_19-23_18_20_dual_strat_wft_ppo_bc_test_random0_test5",

        }
    }

    # plot_runs_training_curves(ppo_bc_model_paths, seeds, save=True)

    set_global_seed(248)
    num_rounds = 100
    ppo_bc_performance = evaluate_two_ppo_models(ppo_bc_model_paths, best_bc_model_paths, num_rounds, seeds, best=True)
    print('SINGLE vs. DUAL ppo_bc_performance', ppo_bc_performance)
    ppo_bc_performance = prepare_nested_default_dict_for_pickle(ppo_bc_performance)
    # save_pickle(ppo_bc_performance, PPO_DATA_DIR + "ppo_bc_models_performance")


def run_two_bc_agents():
    reset_tf()

    # seeds = {
    #     "bc_train": [9456, 1887, 5578, 5987,  516],
    #     "bc_test": [2888, 7424, 7360, 4467,  184]
    # }
    seeds = {
        "bc_train": [9456, 1887, 5578, 5987, 516],
        "bc_test": [9456, 1887, 5578, 5987, 516],
    }

    best_bc_model_paths = {
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

    set_global_seed(248)
    num_rounds = 100
    bc_bc_performance = evaluate_two_bc_models(best_bc_model_paths, num_rounds, seeds, best=True)
    print('BC vs. BC bc_bc_performance', bc_bc_performance)
    # bc_bc_performance  = prepare_nested_default_dict_for_pickle(bc_bc_performance)
    # save_pickle(ppo_bc_performance, PPO_DATA_DIR + "ppo_bc_models_performance")


def play_ppo_vs_ppo(num_games, p1_file, p2_file):
    reset_tf()

    seeds = {
        "train": [9456, 1887, 5578, 5987, 516],
        "test": [2888, 7424, 7360, 4467,  184]
    }

    bc_model_path_for_loading = 'pass_middle_finetune_random3_bc_test_seed5415'

    set_global_seed(248)

    ppo_ppo_performance = []

    agent_bc_test, bc_params = get_bc_agent_from_saved(bc_model_path_for_loading)

    p1_seeds = seeds['train'] if 'train' in p1_file else seeds['test']
    p2_seeds = seeds['train'] if 'train' in p2_file else seeds['test']

    evaluator = AgentEvaluator(mdp_params=bc_params["mdp_params"], env_params=bc_params["env_params"])

    for p1_seed_idx in p1_seeds:
        for p2_seed_idx in p2_seeds:
            p1_ppo_agent, p1_ppo_config = get_ppo_agent(p1_file, p1_seed_idx, best=True)
            p2_ppo_agent, p2_ppo_config = get_ppo_agent(p2_file, p2_seed_idx, best=True)

            # Play two agents against each other.
            ppo_and_ppo = evaluator.evaluate_agent_pair(
                AgentPair(p1_ppo_agent, p2_ppo_agent, allow_duplicate_agents=False), num_games=50, display=False)

            avg_ppo_and_ppo = np.mean(ppo_and_ppo['ep_returns'])
            ppo_ppo_performance.append(avg_ppo_and_ppo)


    mean_reward = np.mean(ppo_ppo_performance)
    std_reward = np.std(ppo_ppo_performance)
    return mean_reward, std_reward


def play_ppo_vs_bc(num_games, p1_file, p2_file):
    reset_tf()

    seeds = {
        "train": [9456, 1887, 5578, 5987, 516],
        "test": [2888, 7424, 7360, 4467, 184]
    }

    p1_seeds = seeds['train'] if 'train' in p1_file else seeds['test']

    set_global_seed(248)

    ppo_bc_performance = []

    p2_agent_bc_test, p2_bc_params = get_bc_agent_from_saved(p2_file)

    p1_ppo_bc_train_path = p1_file

    evaluator = AgentEvaluator(mdp_params=p2_bc_params["mdp_params"], env_params=p2_bc_params["env_params"])

    for seed_idx in p1_seeds:
        p1_agent_ppo_bc_train, p1_ppo_config = get_ppo_agent(p1_ppo_bc_train_path, seed_idx, best=True)

        # How well it generalizes to new agent in simulation?
        ppo_and_bc = evaluator.evaluate_agent_pair(AgentPair(p1_agent_ppo_bc_train, p2_agent_bc_test), num_games=50,
                                                   display=False)

        avg_ppo_and_bc = np.mean(ppo_and_bc['ep_returns'])
        ppo_bc_performance.append(avg_ppo_and_bc)

    mean_reward = np.mean(ppo_bc_performance)
    std_reward = np.std(ppo_bc_performance)
    return mean_reward, std_reward


def play_bc_vs_ppo(num_games, p1_file, p2_file):
    reset_tf()

    seeds = {
        "train": [9456, 1887, 5578, 5987, 516],
        "test": [2888, 7424, 7360, 4467, 184]
    }

    p2_seeds = seeds['train'] if 'train' in p2_file else seeds['test']

    set_global_seed(248)

    bc_ppo_performance = []

    p1_agent_bc_test, p1_bc_params = get_bc_agent_from_saved(p1_file)

    p2_ppo_bc_train_path = p2_file

    evaluator = AgentEvaluator(mdp_params=p1_bc_params["mdp_params"], env_params=p1_bc_params["env_params"])

    for seed_idx in p2_seeds:
        p2_agent_ppo_bc_train, p2_ppo_config = get_ppo_agent(p2_ppo_bc_train_path, seed_idx, best=True)

        # How well it generalizes to new agent in simulation?
        bc_and_ppo = evaluator.evaluate_agent_pair(AgentPair(p1_agent_bc_test, p2_agent_ppo_bc_train),
                                                   num_games=50,
                                                   display=False)

        avg_bc_and_ppo = np.mean(bc_and_ppo['ep_returns'])
        bc_ppo_performance.append(avg_bc_and_ppo)

    mean_reward = np.mean(bc_ppo_performance)
    std_reward = np.std(bc_ppo_performance)

    return mean_reward, std_reward


def play_bc_vs_bc(num_games, p1_file, p2_file):
    reset_tf()

    seeds = {
        "train": [9456, 1887, 5578, 5987, 516],
        "test": [2888, 7424, 7360, 4467, 184]
    }


    set_global_seed(248)

    p1_bc_agent, p1_bc_params = get_bc_agent_from_saved(p1_file)
    p2_bc_agent, p2_bc_params = get_bc_agent_from_saved(p2_file)

    evaluator = AgentEvaluator(mdp_params=p1_bc_params["mdp_params"], env_params=p1_bc_params["env_params"])

    # Play two agents against each other.
    bc_and_bc = evaluator.evaluate_agent_pair(AgentPair(p1_bc_agent, p2_bc_agent, allow_duplicate_agents=False),
                                              num_games=100, display=False)
    mean_reward = np.mean(bc_and_bc['ep_returns'])
    std_reward = np.std(bc_and_bc['ep_returns'])

    return mean_reward, std_reward

def run_all_evaluation_combinations(num_games = 100):
    model_files_dict = {
        'PPO - Strat2(Pass - Middle)': '2021_08_22-02_09_36_pass_middle_ppo_bc_train_random3',
        'PPO - Strat1 / 0(Carry)': '2021_08_22-02_10_36_carry_general_ppo_bc_train_random3',
        'PPO - Strat1(Carry)': '2021_08_22-02_12_05_carry1_ppo_bc_train_random3',
        'PPO - Strat0(Carry)': '2021_08_22-02_11_28_carry0_ppo_bc_train_random3',
        'PPO - Test': '2021_08_22-20_55_15_pass_middle_ppo_bc_test_random3',
        'BC - Train(Strat2)': 'pass_middle_finetune_random3_bc_train_seed5415',
        'BC - Train(Strat1 / 0)': 'carry_general_finetune_random3_bc_train_seed5415',
        'BC - Train(Strat1)': 'carry1_finetune_random3_bc_train_seed5415',
        'BC - Train(Strat0)': 'carry0_finetune_random3_bc_train_seed5415',
        'BC - Test': 'pass_middle_finetune_random3_bc_test_seed5415'
    }

    outcome_dict = {}


    for agent_name_p1 in model_files_dict:
        with open('counter_circuit_compare_outcome_dict1.pickle', 'wb') as handle:
            pickle.dump(outcome_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        for agent_name_p2 in model_files_dict:
            print("RUNNING FOR ", (agent_name_p1, agent_name_p2))
            p1_file = model_files_dict[agent_name_p1]
            p2_file = model_files_dict[agent_name_p2]

            # Play PPO vs. PPO
            if "PPO" in agent_name_p1 and "PPO" in agent_name_p2:
                mean_reward, std_reward = play_ppo_vs_ppo(num_games, p1_file, p2_file)
                outcome_key = (agent_name_p1, agent_name_p2)
                outcome_dict[outcome_key] = (mean_reward, std_reward)


            # Play PPO vs. BC
            if "PPO" in agent_name_p1 and "BC" in agent_name_p2:
                mean_reward, std_reward = play_ppo_vs_bc(num_games, p1_file, p2_file)
                outcome_key = (agent_name_p1, agent_name_p2)
                outcome_dict[outcome_key] = (mean_reward, std_reward)

            # Play BC vs. PPO
            if "BC" in agent_name_p1 and "PPO" in agent_name_p2:
                mean_reward, std_reward = play_bc_vs_ppo(num_games, p1_file, p2_file)
                outcome_key = (agent_name_p1, agent_name_p2)
                outcome_dict[outcome_key] = (mean_reward, std_reward)

            # Play BC vs. BC
            if "BC" in agent_name_p1 and "BC" in agent_name_p2:
                mean_reward, std_reward = play_bc_vs_bc(num_games, p1_file, p2_file)
                outcome_key = (agent_name_p1, agent_name_p2)
                outcome_dict[outcome_key] = (mean_reward, std_reward)




    print('outcome_dict = ', outcome_dict)
    return outcome_dict

if __name__ == "__main__":


    # print('\n\n\n RUNNING COMPARISON AGENTS.................')
    # outcome_dict = run_all_evaluation_combinations(num_games = 100)

    with open('counter_circuit_compare_outcome_dict1.pickle', 'rb') as handle:
        outcome = pickle.load(handle)

    print(outcome)
    with open('cc_outcome.txt', 'w') as f:
        print(outcome, file=f)