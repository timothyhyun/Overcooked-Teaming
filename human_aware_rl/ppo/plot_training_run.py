import gym, time, os, seaborn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from memory_profiler import profile

from sacred import Experiment
from sacred.observers import FileStorageObserver
from tensorflow.saved_model import simple_save

# import wandb
# from wandb.keras import WandbCallback
# #
# # wandb.login() #68e6971049b2c9550c19b9fbfdadf55bc556d2f5
# # wandb.init(project="Test-project-ppo-strat-specific")
# wandb.init(project = "Test-project-ppo-strat-specific", sync_tensorboard=True)

# logging code
# run = wandb.init(project="ppo-strat")

PPO_DATA_DIR = 'data/ppo_runs/'

ex = Experiment('PPO')
ex.observers.append(FileStorageObserver.create(PPO_DATA_DIR + 'ppo_exp'))

import sys

sys.path.insert(0, "../../")

from overcooked_ai_py.utils import load_pickle, save_pickle, load_dict_from_file, profile
from overcooked_ai_py.agents.agent import RandomAgent, GreedyHumanModel, AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.planning.planners import NO_COUNTERS_PARAMS, MediumLevelPlanner
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from human_aware_rl.baselines_utils import get_vectorized_gym_env, create_model, update_model, save_baselines_model, \
    load_baselines_model, get_agent_from_saved_model
from human_aware_rl.utils import create_dir_if_not_exists, reset_tf, delete_dir_if_exists, set_global_seed
from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved, DEFAULT_ENV_PARAMS, BC_SAVE_DIR
from human_aware_rl.experiments.bc_experiments import BEST_BC_MODELS_PATH


# PARAMS
@ex.config
def my_config():
    ##################
    # GENERAL PARAMS #
    ##################

    TIMESTAMP_DIR = True
    EX_NAME = "undefined_name"

    if TIMESTAMP_DIR:
        SAVE_DIR = PPO_DATA_DIR + time.strftime('%Y_%m_%d-%H_%M_%S_') + EX_NAME + "/"
    else:
        SAVE_DIR = PPO_DATA_DIR + EX_NAME + "/"

    print("Saving data to ", SAVE_DIR)

    RUN_TYPE = "ppo"

    STRATEGY_INDEX = 0

    # Reduce parameters to be able to run locally to test for simple bugs
    LOCAL_TESTING = False

    # Choice among: bc_train, bc_test, sp, hm, rnd
    OTHER_AGENT_TYPE = "bc_train"

    # Human model params, only relevant if OTHER_AGENT_TYPE is "hm"
    HM_PARAMS = [True, 0.3]

    # GPU id to use
    GPU_ID = 1

    # List of seeds to run
    SEEDS = [0]

    # Number of parallel environments used for simulating rollouts
    sim_threads = 30 if not LOCAL_TESTING else 2

    # Threshold for sparse reward before saving the best model
    SAVE_BEST_THRESH = 50

    # Every `VIZ_FREQUENCY` gradient steps, display the first 100 steps of a rollout of the agents
    VIZ_FREQUENCY = 50 if not LOCAL_TESTING else 10

    ##############
    # PPO PARAMS #
    ##############

    # Total environment timesteps for the PPO run
    PPO_RUN_TOT_TIMESTEPS = 5e6 if not LOCAL_TESTING else 10000

    # How many environment timesteps will be simulated (across all environments)
    # for one set of gradient updates. Is divided equally across environments
    TOTAL_BATCH_SIZE = 12000 if not LOCAL_TESTING else 800

    # Number of minibatches we divide up each batch into before
    # performing gradient steps
    MINIBATCHES = 6 if not LOCAL_TESTING else 1

    # Calculating `batch size` as defined in baselines
    BATCH_SIZE = TOTAL_BATCH_SIZE // sim_threads

    # Number of gradient steps to perform on each mini-batch
    STEPS_PER_UPDATE = 8 if not LOCAL_TESTING else 1

    # Learning rate
    LR = 1e-3

    # Factor by which to reduce learning rate over training
    LR_ANNEALING = 1

    # Entropy bonus coefficient
    ENTROPY = 0.1

    # Value function coefficient
    VF_COEF = 0.1

    # Gamma discounting factor
    GAMMA = 0.99

    # Lambda advantage discounting factor
    LAM = 0.98

    # Max gradient norm
    MAX_GRAD_NORM = 0.1

    # PPO clipping factor
    CLIPPING = 0.05

    # None is default value that does no schedule whatsoever
    # [x, y] defines the beginning of non-self-play trajectories
    SELF_PLAY_HORIZON = None

    # 0 is default value that does no annealing
    REW_SHAPING_HORIZON = 0

    # Whether mixing of self play policies
    # happens on a trajectory or on a single-timestep level
    # Recommended to keep to true
    TRAJECTORY_SELF_PLAY = True

    ##################
    # NETWORK PARAMS #
    ##################

    # Network type used
    NETWORK_TYPE = "conv_and_mlp"

    # Network params
    NUM_HIDDEN_LAYERS = 3
    SIZE_HIDDEN_LAYERS = 64
    NUM_FILTERS = 25
    NUM_CONV_LAYERS = 3

    ##################
    # MDP/ENV PARAMS #
    ##################

    # Mdp params
    layout_name = None
    start_order_list = None

    # STRATEGY_REWARD_PARAMS_7 = {
    #     'simple': {
    #         0:  [2.70252993e-01, 1.96315895e-01, 7.52119554e-02, 3.21244143e-01, 9.92909390e-09, 1.32576967e-01, 4.39803668e-03],
    #         1:  [7.87017954e-02, 1.67897174e-01, 1.13180689e-01, 5.51662015e-01, 9.02573457e-09, 8.61972170e-02, 2.36110054e-03],
    #         2: [1.14521552e-01, 1.96694494e-01, 1.34228192e-01, 4.82998338e-01, 5.28267755e-09, 7.02746012e-02, 1.28281764e-03],
    #         3: [1.23943402e-01, 2.70422208e-01, 4.88269705e-02, 4.50702552e-01, 2.21933352e-07, 9.76528287e-02, 8.45181651e-03],
    #     },
    #     'random1': {
    #         0:  [0.26971481, 0.37027598, 0.10447312, 0.10448251, 0.0694081, 0.07847855, 0.00316694],
    #         1:  [0.12336561, 0.29372795, 0.14980191, 0.17623721, 0.06873219, 0.18504866, 0.00308648],
    #     },
    #     'unident_s': {
    #         0: [0.18829796, 0.36893755, 0.15159763, 0.14966015, 0.03527472, 0.10499902, 0.00123298],
    #         1: [0.15212459, 0.31595109, 0.12872085, 0.19746945, 0.03356981, 0.16821472, 0.00394949],
    #     },
    #     "random0": {
    #         0:  [0.11947069, 0.27646117, 0.1184835, 0.15699054, 0.05435423, 0.16982616, 0.10441369],
    #         1: [0.14300781, 0.22740592, 0.1062791, 0.13988202, 0.1340991, 0.1547298, 0.09459625],
    #     },
    #     "random3": {
    #         0: [0.38372211, 0.29566231, 0.15073883, 0.05588366, 0.00192108, 0.10201446, 0.01005756],
    #         1: [0.30387411, 0.41368073, 0.09275247, 0.11054466, 0.03021337, 0.03886827, 0.01006639],
    #         2: [0.23454106, 0.40847999, 0.14816318, 0.12144665, 0.02265959, 0.05814603, 0.0065635],
    #         3: [0.22187374, 0.38093024, 0.14346863, 0.08025317, 0.07614822, 0.09586659, 0.00145941],
    #     },
    # }

    STRATEGY_REWARD_PARAMS = {
        'simple': {
            0: [2.70252993e-01, 1.96315895e-01, 7.52119554e-02, 3.21244143e-01, 9.92909390e-09, 1.32576967e-01],
            1: [7.87017954e-02, 1.67897174e-01, 1.13180689e-01, 5.51662015e-01, 9.02573457e-09, 8.61972170e-02],
            2: [1.14521552e-01, 1.96694494e-01, 1.34228192e-01, 4.82998338e-01, 5.28267755e-09, 7.02746012e-02],
            3: [1.23943402e-01, 2.70422208e-01, 4.88269705e-02, 4.50702552e-01, 2.21933352e-07, 9.76528287e-02],
        },
        'random1': {
            0: [0.26971481, 0.37027598, 0.10447312, 0.10448251, 0.0694081, 0.07847855],
            1: [0.12336561, 0.29372795, 0.14980191, 0.17623721, 0.06873219, 0.18504866],
        },
        'unident_s': {
            0: [0.18829796, 0.36893755, 0.15159763, 0.14966015, 0.03527472, 0.10499902],
            1: [0.15212459, 0.31595109, 0.12872085, 0.19746945, 0.03356981, 0.16821472],
        },
        "random0": {
            0: [0.11947069, 0.27646117, 0.1184835, 0.15699054, 0.05435423, 0.16982616],
            1: [0.14300781, 0.22740592, 0.1062791, 0.13988202, 0.1340991, 0.1547298],
        },
        "random3": {
            0: [0.38372211, 0.29566231, 0.15073883, 0.05588366, 0.00192108, 0.10201446],
            1: [0.30387411, 0.41368073, 0.09275247, 0.11054466, 0.03021337, 0.03886827],
            2: [0.23454106, 0.40847999, 0.14816318, 0.12144665, 0.02265959, 0.05814603],
            3: [0.22187374, 0.38093024, 0.14346863, 0.08025317, 0.07614822, 0.09586659],
        },
    }
    # rew_shaping_params = {
    #     "PLACEMENT_IN_POT_REW": 3,
    #     "DISH_PICKUP_REWARD": 3,
    #     "SOUP_PICKUP_REWARD": 5,
    #     "DISH_DISP_DISTANCE_REW": 0,
    #     "POT_DISTANCE_REW": 0,
    #     "SOUP_DISTANCE_REW": 0,
    # }
    # rew_shaping_params = {
    #     "PLACEMENT_IN_POT_REW": 0,
    #     "DISH_PICKUP_REWARD": 0,
    #     "SOUP_PICKUP_REWARD": 0,
    #     "DISH_DISP_DISTANCE_REW": 0,
    #     "POT_DISTANCE_REW": 0,
    #     "SOUP_DISTANCE_REW": 0,
    # }
    # rew_shaping_params = {
    #     "ONION_IN_EMPTY_POT_REWARD": 0,
    #     "ONION_IN_PARTIAL_POT_REWARD": 0,
    #     "DISH_PICKUP_REWARD": 0,
    #     "SOUP_PICKUP_FROM_READY_POT_REWARD": 0,
    #     "BOTH_POTS_FULL_REWARD": 0,
    #     "SERVE_SOUP_REWARD": 0,
    #     "SHARED_COUNTER_REWARD": 0,
    # }
    # print("STRATEGY_INDEX", STRATEGY_INDEX)
    # STRATEGY_INDEX = int(STRATEGY_INDEX)
    # print("layout_name", layout_name)
    # print("test", STRATEGY_REWARD_PARAMS[layout_name][STRATEGY_INDEX])
    rew_shaping_params = {
        "ONION_IN_EMPTY_POT_REWARD": STRATEGY_REWARD_PARAMS[layout_name][STRATEGY_INDEX][0],
        "ONION_IN_PARTIAL_POT_REWARD": STRATEGY_REWARD_PARAMS[layout_name][STRATEGY_INDEX][1],
        "DISH_PICKUP_REWARD": STRATEGY_REWARD_PARAMS[layout_name][STRATEGY_INDEX][2],
        "SOUP_PICKUP_FROM_READY_POT_REWARD": STRATEGY_REWARD_PARAMS[layout_name][STRATEGY_INDEX][3],
        "BOTH_POTS_FULL_REWARD": STRATEGY_REWARD_PARAMS[layout_name][STRATEGY_INDEX][4],
        "SERVE_SOUP_REWARD": STRATEGY_REWARD_PARAMS[layout_name][STRATEGY_INDEX][5],
        # "SHARED_COUNTER_REWARD": STRATEGY_REWARD_PARAMS[layout_name][STRATEGY_INDEX][6],
    }
    # for elem in STRATEGY_REWARD_PARAMS[layout_name][STRATEGY_INDEX]:
    #     rew_shaping_params[]

    # Env params
    horizon = 400

    # For non fixed MDPs
    mdp_generation_params = {
        "padded_mdp_shape": (11, 7),
        "mdp_shape_fn": ([5, 11], [5, 7]),
        "prop_empty_fn": [0.6, 1],
        "prop_feats_fn": [0, 0.6]
    }

    # Approximate info
    GRAD_UPDATES_PER_AGENT = STEPS_PER_UPDATE * MINIBATCHES * (PPO_RUN_TOT_TIMESTEPS // TOTAL_BATCH_SIZE)
    print("Grad updates per agent", GRAD_UPDATES_PER_AGENT)

    params = {
        "STRATEGY_INDEX": STRATEGY_INDEX,
        "RUN_TYPE": RUN_TYPE,
        "SEEDS": SEEDS,
        "LOCAL_TESTING": LOCAL_TESTING,
        "EX_NAME": EX_NAME,
        "SAVE_DIR": SAVE_DIR,
        "GPU_ID": GPU_ID,
        "PPO_RUN_TOT_TIMESTEPS": PPO_RUN_TOT_TIMESTEPS,
        "mdp_params": {
            "layout_name": layout_name,
            "start_order_list": start_order_list,
            "rew_shaping_params": rew_shaping_params
        },
        "env_params": {
            "horizon": horizon
        },
        "mdp_generation_params": mdp_generation_params,
        "ENTROPY": ENTROPY,
        "GAMMA": GAMMA,
        "sim_threads": sim_threads,
        "TOTAL_BATCH_SIZE": TOTAL_BATCH_SIZE,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_GRAD_NORM": MAX_GRAD_NORM,
        "LR": LR,
        "LR_ANNEALING": LR_ANNEALING,
        "VF_COEF": VF_COEF,
        "STEPS_PER_UPDATE": STEPS_PER_UPDATE,
        "MINIBATCHES": MINIBATCHES,
        "CLIPPING": CLIPPING,
        "LAM": LAM,
        "SELF_PLAY_HORIZON": SELF_PLAY_HORIZON,
        "REW_SHAPING_HORIZON": REW_SHAPING_HORIZON,
        "OTHER_AGENT_TYPE": OTHER_AGENT_TYPE,
        "HM_PARAMS": HM_PARAMS,
        "NUM_HIDDEN_LAYERS": NUM_HIDDEN_LAYERS,
        "SIZE_HIDDEN_LAYERS": SIZE_HIDDEN_LAYERS,
        "NUM_FILTERS": NUM_FILTERS,
        "NUM_CONV_LAYERS": NUM_CONV_LAYERS,
        "NETWORK_TYPE": NETWORK_TYPE,
        "SAVE_BEST_THRESH": SAVE_BEST_THRESH,
        "TRAJECTORY_SELF_PLAY": TRAJECTORY_SELF_PLAY,
        "VIZ_FREQUENCY": VIZ_FREQUENCY,
        "grad_updates_per_agent": GRAD_UPDATES_PER_AGENT
    }


def save_ppo_model(model, save_folder):
    delete_dir_if_exists(save_folder, verbose=True)
    simple_save(
        tf.get_default_session(),
        save_folder,
        inputs={"obs": model.act_model.X},
        outputs={
            "action": model.act_model.action,
            "value": model.act_model.vf,
            "action_probs": model.act_model.action_probs
        }
    )


def configure_other_agent(params, gym_env, mlp, mdp):
    # The other agent is a greedy human-like agent but nothing is trained. Agent that at each step selects a medium level action corresponding
    #     to the most intuitively high-priority thing to do.

    if params["OTHER_AGENT_TYPE"] == "hm":
        hl_br, hl_temp, ll_br, ll_temp = params["HM_PARAMS"]
        agent = GreedyHumanModel(mlp, hl_boltzmann_rational=hl_br, hl_temp=hl_temp, ll_boltzmann_rational=ll_br,
                                 ll_temp=ll_temp)
        gym_env.use_action_method = True


    # If the other agent is the behaviorally cloned human-proxy model

    elif params["OTHER_AGENT_TYPE"][:2] == "bc":
        # best_bc_model_paths = load_pickle(BEST_BC_MODELS_PATH)
        # print('best_bc_model_paths', best_bc_model_paths)
        # best_bc_model_paths = {
        #     'train': {
        #         'simple': 'simple_bc_train_seed3',
        #         'random1': 'random1_bc_train_seed0',
        #         'unident_s': 'aa_strat3_finetune_unident_s_bc_train_seed5415',
        #         # 'random0': 'single_pot_finetune_random0_bc_train_seed5415',
        #         'random0': 'dual_pot_finetune_random0_bc_train_seed5415',
        #         'random3': 'carry1_finetune_random3_bc_train_seed5415'
        #     },
        #      'test': {
        #          'simple': 'simple_bc_test_seed2',
        #          'random1': 'random1_bc_test_seed2',
        #          'unident_s': 'aa_strat3_finetune_unident_s_bc_test_seed5415',
        #          # 'random0': 'single_pot_finetune_random0_bc_test_seed5415',
        #          'random0': 'dual_pot_finetune_random0_bc_test_seed5415',
        #          'random3': 'carry1_finetune_random3_bc_test_seed5415'
        #      }
        # }
        # initial_best_bc_model_paths = {
        #     'train': {
        #         'simple': 'simple_bc_train_seed3',
        #         'random1': 'random1_bc_train_seed0',
        #         'unident_s': 'aa_strat3_finetune_unident_s_bc_train_seed5415',
        #         # 'random0': 'single_pot_finetune_random0_bc_test_seed5415',
        #         'random0': 'dual_pot_finetune_random0_bc_test_seed5415',
        #         # 'random0': 'single_pot_finetune_random0_bc_train_seed5415',
        #         # 'random0': 'bc_train_fixed_strat_DP_DP_random0_bc_train_seed5415',
        #         'random3': 'carry1_finetune_random3_bc_train_seed5415'
        #     },
        #     'test': {
        #         'simple': 'simple_bc_test_seed2',
        #         'random1': 'random1_bc_test_seed2',
        #         'unident_s': 'aa_strat3_finetune_unident_s_bc_test_seed5415',
        #
        #         'random0': 'single_pot_finetune_random0_bc_test_seed5415',
        #         # 'random0': 'bc_train_fixed_strat_SP_SP_random0_bc_train_seed5415',
        #         'random3': 'carry1_finetune_random3_bc_test_seed5415'
        #     }
        # }

        best_bc_model_paths = {
            'train': {
                'simple': {
                    0: "simple_strat0_finetune_seed5415",
                    1: "simple_strat1_finetune_seed5415",
                    2: "simple_strat2_finetune_seed5415",
                    3: "simple_strat3_finetune_seed5415",
                },
                'random1': {
                    0: "random1_strat0_finetune_seed5415",
                    1: "random1_strat1_finetune_seed5415",
                },
                'unident_s': {
                    0: "unident_s_strat0_finetune_seed5415",
                    1: "unident_s_strat1_finetune_seed5415",
                },
                "random0": {
                    0: "random0_strat0_finetune_seed5415",
                    1: "random0_strat1_finetune_seed5415",
                },
                "random3": {
                    0: "random3_strat0_finetune_seed5415",
                    1: "random3_strat1_finetune_seed5415",
                    2: "random3_strat2_finetune_seed5415",
                    3: "random3_strat3_finetune_seed5415",
                },
            },

        }

        # best_bc_model_paths = load_pickle(BEST_BC_MODELS_PATH) # uncomment to change
        print('\n\n\n\nbest_bc_model_paths', best_bc_model_paths)

        if params["OTHER_AGENT_TYPE"] == "bc_train":
            bc_model_path = best_bc_model_paths["train"][mdp.layout_name][params["STRATEGY_INDEX"]]
        elif params["OTHER_AGENT_TYPE"] == "bc_test":
            bc_model_path = best_bc_model_paths["train"][mdp.layout_name][params["STRATEGY_INDEX"]]
        else:
            raise ValueError("Other agent type must be bc train or bc test")

        print("LOADING BC MODEL FROM: {}".format(bc_model_path))
        agent, bc_params = get_bc_agent_from_saved(bc_model_path)
        gym_env.use_action_method = True
        # Make sure environment params are the same in PPO as in the BC model
        for k, v in bc_params["env_params"].items():
            assert v == params["env_params"][k], "{} did not match. env_params: {} \t PPO params: {}".format(k, v,
                                                                                                             params[k])
        for k, v in bc_params["mdp_params"].items():
            assert v == params["mdp_params"][k], "{} did not match. mdp_params: {} \t PPO params: {}".format(k, v,
                                                                                                             params[k])

    elif params["OTHER_AGENT_TYPE"] == "rnd":
        agent = RandomAgent()

    elif params["OTHER_AGENT_TYPE"] == "sp":
        gym_env.self_play_randomization = 1

    else:
        raise ValueError("unknown type of agent to match with")

    if not params["OTHER_AGENT_TYPE"] == "sp":
        assert mlp.mdp == mdp
        agent.set_mdp(mdp)
        gym_env.other_agent = agent


def load_training_data(run_name, seeds=None):
    run_dir = '../experiments/' + PPO_DATA_DIR + run_name + "/"
    config = load_pickle(run_dir + "config")

    # To add backwards compatibility
    if seeds is None:
        if "NUM_SEEDS" in config.keys():
            seeds = list(range(min(config["NUM_SEEDS"], 5)))
        else:
            seeds = config["SEEDS"]

    train_infos = []
    for seed in seeds:
        train_info = load_pickle(run_dir + "seed{}/training_info".format(seed))
        train_infos.append(train_info)

    return train_infos, config


def get_ppo_agent(save_dir, seed=0, best=False):
    # save_dir = 'ppo/' + PPO_DATA_DIR + save_dir + '/seed{}'.format(seed)
    save_dir = '../experiments/results_forte2/' + PPO_DATA_DIR + save_dir + '/seed{}'.format(seed)
    config = load_pickle(save_dir + '/config')
    if best:
        agent = get_agent_from_saved_model(save_dir + "/best", config["sim_threads"])
    else:
        agent = get_agent_from_saved_model(save_dir + "/ppo_agent", config["sim_threads"])
    return agent, config


def match_ppo_with_other_agent(save_dir, other_agent, n=1, display=False):
    agent, agent_eval = get_ppo_agent(save_dir)
    ap0 = AgentPair(agent, other_agent)
    agent_eval.evaluate_agent_pair(ap0, display=display, num_games=n)

    # Sketch switch
    ap1 = AgentPair(other_agent, agent)
    agent_eval.evaluate_agent_pair(ap1, display=display, num_games=n)


def plot_ppo_run(name, sparse=False, limit=None, print_config=False, seeds=None, single=False):
    from collections import defaultdict
    seeds = [9456]
    train_infos, config = load_training_data(name, seeds)

    if print_config:
        print(config)

    if limit is None:
        limit = config["PPO_RUN_TOT_TIMESTEPS"]

    num_datapoints = len(train_infos[0]['eprewmean'])

    prop_data = limit / config["PPO_RUN_TOT_TIMESTEPS"]
    ciel_data_idx = int(num_datapoints * prop_data)

    datas = []
    for seed_num, info in enumerate(train_infos):
        info['xs'] = config["TOTAL_BATCH_SIZE"] * np.array(range(1, ciel_data_idx + 1))
        if single:
            plt.plot(info['xs'], info["ep_sparse_rew_mean"][:ciel_data_idx], alpha=1, label="Sparse{}".format(seed_num))
        datas.append(info["ep_sparse_rew_mean"][:ciel_data_idx])
    if not single:
        seaborn.tsplot(time=info['xs'], data=datas)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    if single:
        plt.legend()

    plt.xlabel("Timestep")
    plt.ylabel("Sparse Reward")
    plt.title("SPARSE: "+name)
    plt.savefig("training_imgs/sparse_training_"+name+".png")
    plt.close()


def get_weight_for_serve(layout_name, strategy_number):
    STRATEGY_REWARD_PARAMS = {
        'simple': {
            0: [2.70252993e-01, 1.96315895e-01, 7.52119554e-02, 3.21244143e-01, 9.92909390e-09, 1.32576967e-01],
            1: [7.87017954e-02, 1.67897174e-01, 1.13180689e-01, 5.51662015e-01, 9.02573457e-09, 8.61972170e-02],
            2: [1.14521552e-01, 1.96694494e-01, 1.34228192e-01, 4.82998338e-01, 5.28267755e-09, 7.02746012e-02],
            3: [1.23943402e-01, 2.70422208e-01, 4.88269705e-02, 4.50702552e-01, 2.21933352e-07, 9.76528287e-02],
        },
        'random1': {
            0: [0.26971481, 0.37027598, 0.10447312, 0.10448251, 0.0694081, 0.07847855],
            1: [0.12336561, 0.29372795, 0.14980191, 0.17623721, 0.06873219, 0.18504866],
        },
        'unident': {
            0: [0.18829796, 0.36893755, 0.15159763, 0.14966015, 0.03527472, 0.10499902],
            1: [0.15212459, 0.31595109, 0.12872085, 0.19746945, 0.03356981, 0.16821472],
        },
        "random0": {
            0: [0.11947069, 0.27646117, 0.1184835, 0.15699054, 0.05435423, 0.16982616],
            1: [0.14300781, 0.22740592, 0.1062791, 0.13988202, 0.1340991, 0.1547298],
        },
        "random3": {
            0: [0.38372211, 0.29566231, 0.15073883, 0.05588366, 0.00192108, 0.10201446],
            1: [0.30387411, 0.41368073, 0.09275247, 0.11054466, 0.03021337, 0.03886827],
            2: [0.23454106, 0.40847999, 0.14816318, 0.12144665, 0.02265959, 0.05814603],
            3: [0.22187374, 0.38093024, 0.14346863, 0.08025317, 0.07614822, 0.09586659],
        },
    }
    select_weights = STRATEGY_REWARD_PARAMS[layout_name][strategy_number]
    scale = 3 / select_weights[0]
    weight_for_serve = select_weights[5] * scale

    # weight_for_serve = 5
    print("weight_for_serve", weight_for_serve)

    # true point value = N * W * (20/W)
    # weight_for_serve = 20

    return weight_for_serve

def sparse_plot_ppo_run(name, sparse=False, limit=None, print_config=False, seeds=None, single=False):
    from collections import defaultdict
    seeds = [9456, 1887, 5578, 5987, 516]

    layout_name = name.split("_")[2]
    strategy_number = int(name.split("_")[3].split("s")[1])
    weight_for_serve = get_weight_for_serve(layout_name, strategy_number)

    train_infos, config = load_training_data(name, seeds)

    if print_config:
        print(config)

    if limit is None:
        limit = config["PPO_RUN_TOT_TIMESTEPS"]

    num_datapoints = len(train_infos[0]['eprewmean'])

    prop_data = limit / config["PPO_RUN_TOT_TIMESTEPS"]
    ciel_data_idx = int(num_datapoints * prop_data)

    datas = []
    for seed_num, info in enumerate(train_infos):
        info['xs'] = config["TOTAL_BATCH_SIZE"] * np.array(range(1, ciel_data_idx + 1))
        if single:
            plt.plot(info['xs'], info["ep_sparse_rew_mean"][:ciel_data_idx], alpha=1, label="Sparse{}".format(seed_num))
        datas.append(np.array(info["ep_sparse_rew_mean"][:ciel_data_idx])/weight_for_serve)
    if not single:
        seaborn.tsplot(time=info['xs'], data=datas)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    if single:
        plt.legend()

    plt.xlabel("Timestep")
    plt.ylabel("Number of Orders Served in 60s")
    plt.title("Number of Orders for Env: " + layout_name)
    plt.savefig("training_imgs/sparse_training2_"+name+".png")
    plt.close()

def strategy_plot_ppo_run(sparse=False, limit=None, print_config=False, seeds=None, single=True):
    from collections import defaultdict
    # seeds = [9456, 1887, 5578, 5987, 516]
    seeds = [516]



    all_models = {
        "random0": ['STRATEXP_TEST2_random0_s0_weights_ppo_bc_train',
        'STRATEXP_TEST2_random0_s1_weights_ppo_bc_train',
                    "2021_07_21-09_02_08_ppo_bc_train_random0_test1"],

        "random1": ['STRATEXP_TEST2_random1_s0_weights_ppo_bc_train',
        'STRATEXP_TEST2_random1_s1_weights_ppo_bc_train',
                    "2021_08_15-02_55_36_ppo_bc_train_random1_REPLICATE1"],

        "random3": ['STRATEXP_TEST2_random3_s0_weights_ppo_bc_train',
        'STRATEXP_TEST2_random3_s1_weights_ppo_bc_train',
        'STRATEXP_TEST2_random3_s2_weights_ppo_bc_train',
        'STRATEXP_TEST2_random3_s3_weights_ppo_bc_train',
                    "2021_12_17-15_59_55_ppo_bc_train_random3_REPLICATE1"],

        "simple": ['STRATEXP_TEST2_simple_s0_weights_ppo_bc_train',
        'STRATEXP_TEST2_simple_s1_weights_ppo_bc_train',
        'STRATEXP_TEST2_simple_s2_weights_ppo_bc_train',
        'STRATEXP_TEST2_simple_s3_weights_ppo_bc_train',
                   "2021_07_31-12_51_51_ppo_bc_train_simple_REPLICATE1"],

        "unident": [ 'STRATEXP_TEST2_unident_s0_weights_ppo_bc_train',
        'STRATEXP_TEST2_unident_s1_weights_ppo_bc_train',
                     "2021_08_03-11_45_12_ppo_bc_train_unident_s_REPLICATE1"],



    }


    for layout_name in all_models:
        strategy_files = all_models[layout_name]
        for name in strategy_files:
            if "STRATEXP" not in name:
                strategy_number = "baseline PPO_BC"
                weight_for_serve = 20
            else:
                strategy_number = int(name.split("_")[3].split("s")[1])
                weight_for_serve = get_weight_for_serve(layout_name, strategy_number)


            # layout_name = name.split("_")[2]
            # strategy_number = int(name.split("_")[3].split("s")[1])
            # weight_for_serve = get_weight_for_serve(layout_name, strategy_number)
            used_seeds = seeds
            if "STRATEXP" not in name:
                used_seeds = [9456]

            train_infos, config = load_training_data(name, used_seeds)
            batch_size = config["BATCH_SIZE"]
            print(f"{layout_name} batch size = {batch_size}")

            if print_config:
                print(config)

            if limit is None:
                limit = config["PPO_RUN_TOT_TIMESTEPS"]

            num_datapoints = len(train_infos[0]['eprewmean'])

            prop_data = limit / config["PPO_RUN_TOT_TIMESTEPS"]
            ciel_data_idx = int(num_datapoints * prop_data)
            print("ciel_data_idx", ciel_data_idx)
            if layout_name == "simple":
                ciel_data_idx = 666


            datas = []
            for seed_num, info in enumerate(train_infos):
                info['xs'] = config["TOTAL_BATCH_SIZE"] * np.array(range(1, ciel_data_idx + 1))
                print("y data", len(info["ep_sparse_rew_mean"][:]))
                if single:
                    if "STRATEXP" not in name:
                        plt.plot(info['xs'], np.array(info["ep_sparse_rew_mean"][:ciel_data_idx]) *20 / weight_for_serve,
                                 alpha=1,
                                 label="Baseline: PPO BC")
                    else:
                        plt.plot(info['xs'], np.array(info["ep_sparse_rew_mean"][:ciel_data_idx]) *20/ weight_for_serve, alpha=1,
                                 label="Strategy: {}_seed{}".format(strategy_number, used_seeds[seed_num]))
                    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                    # try:
                    #     plt.plot(info['xs'], info["ep_sparse_rew_mean"][:ciel_data_idx], alpha=1,
                    #              label="Strategy: {}_seed{}".format(strategy_number, seeds[seed_num]))
                    #     plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                    #     break
                    # except:
                    #     continue
                datas.append(np.array(info["ep_sparse_rew_mean"][:ciel_data_idx]) / weight_for_serve)
            if not single:
                seaborn.tsplot(time=info['xs'], data=datas)
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))


            # if single:
        plt.legend()


        plt.xlabel("Timestep")
        plt.ylabel("Number of Orders Served in 60s")
        plt.title("Number of Orders for Each Strategy: "+layout_name)
        plt.savefig("training_imgs/sparse_training4_"+layout_name+".png")
        plt.close()



def dense_plot_ppo_run(name, sparse=False, limit=None, print_config=False, seeds=None, single=False):
    from collections import defaultdict
    seeds = [9456]
    train_infos, config = load_training_data(name, seeds)

    if print_config:
        print(config)

    if limit is None:
        limit = config["PPO_RUN_TOT_TIMESTEPS"]

    num_datapoints = len(train_infos[0]['eprewmean'])

    prop_data = limit / config["PPO_RUN_TOT_TIMESTEPS"]
    ciel_data_idx = int(num_datapoints * prop_data)

    datas = []
    for seed_num, info in enumerate(train_infos):
        info['xs'] = config["TOTAL_BATCH_SIZE"] * np.array(range(1, ciel_data_idx + 1))
        if single:
            plt.plot(info['xs'], info["ep_dense_rew_mean"][:ciel_data_idx], alpha=1, label="Dense{}".format(seed_num))
        datas.append(info["ep_dense_rew_mean"][:ciel_data_idx])
    if not single:
        seaborn.tsplot(time=info['xs'], data=datas)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    if single:
        plt.legend()

    plt.xlabel("Timestep")
    plt.ylabel("Dense Reward")
    plt.title("DENSE: "+name)
    plt.savefig("training_imgs/dense_training_"+name+".png")
    plt.close()

if __name__ == "__main__":
    strategy_plot_ppo_run()
    # all_models = [
    #     'STRATEXP_TEST2_random0_s0_weights_ppo_bc_train',
    #     'STRATEXP_TEST2_random0_s1_weights_ppo_bc_train',
    #
    #     'STRATEXP_TEST2_random1_s0_weights_ppo_bc_train',
    #     'STRATEXP_TEST2_random1_s1_weights_ppo_bc_train',
    #
    #     'STRATEXP_TEST2_random3_s0_weights_ppo_bc_train',
    #     'STRATEXP_TEST2_random3_s1_weights_ppo_bc_train',
    #     'STRATEXP_TEST2_random3_s2_weights_ppo_bc_train',
    #     'STRATEXP_TEST2_random3_s3_weights_ppo_bc_train',
    #
    #     'STRATEXP_TEST2_simple_s0_weights_ppo_bc_train',
    #     'STRATEXP_TEST2_simple_s1_weights_ppo_bc_train',
    #     'STRATEXP_TEST2_simple_s2_weights_ppo_bc_train',
    #     'STRATEXP_TEST2_simple_s3_weights_ppo_bc_train',
    #
    #     'STRATEXP_TEST2_unident_s0_weights_ppo_bc_train',
    #     'STRATEXP_TEST2_unident_s1_weights_ppo_bc_train',
    #
    # ]
    # for name in all_models:
    #     sparse_plot_ppo_run(name)


# @ex.automain
# @profile
def ppo_run(params):
    # print("\n\n\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    create_dir_if_not_exists(params["SAVE_DIR"])
    save_pickle(params, params["SAVE_DIR"] + "config")

    #############
    # PPO SETUP #
    #############

    train_infos = []

    for seed in params["SEEDS"]:
        reset_tf()
        set_global_seed(seed)

        curr_seed_dir = params["SAVE_DIR"] + "seed" + str(seed) + "/"
        create_dir_if_not_exists(curr_seed_dir)

        save_pickle(params, curr_seed_dir + "config")

        print("Creating env with params", params)
        # Configure mdp

        mdp = OvercookedGridworld.from_layout_name(**params["mdp_params"])
        env = OvercookedEnv(mdp, **params["env_params"])
        mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=True)

        # Configure gym env
        gym_env = get_vectorized_gym_env(
            env, 'Overcooked-v0', featurize_fn=lambda x: mdp.lossless_state_encoding(x), **params
        )
        gym_env.self_play_randomization = 0 if params["SELF_PLAY_HORIZON"] is None else 1
        gym_env.trajectory_sp = params["TRAJECTORY_SELF_PLAY"]
        gym_env.update_reward_shaping_param(1 if params["mdp_params"]["rew_shaping_params"] != 0 else 0)

        configure_other_agent(params, gym_env, mlp, mdp)

        # Create model
        with tf.device('/device:GPU:{}'.format(params["GPU_ID"])):
            model = create_model(gym_env, "ppo_agent", **params)

        # Train model
        params["CURR_SEED"] = seed
        train_info = update_model(gym_env, model, **params)

        # Save model
        save_ppo_model(model, curr_seed_dir + model.agent_name)
        print("Saved training info at", curr_seed_dir + "training_info")
        save_pickle(train_info, curr_seed_dir + "training_info")
        train_infos.append(train_info)

    return train_infos


