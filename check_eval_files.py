import numpy as np
import pickle as pkl



if __name__ == "__main__":
    # filename = 'human_aware_rl/data/bc_runs/best_bc_models_performance.pickle'
    # file = open(filename, 'rb')
    # best_models_perf = pkl.load(file)
    # file.close()
    # print('best_models_perf = ', best_models_perf)

    # filename = 'human_aware_rl/data/bc_runs/bc_models_all_evaluations.pickle'
    # file = open(filename, 'rb')
    # best_models_evals = pkl.load(file)
    # file.close()
    # print('best_models_evals = ', best_models_evals)

    # filename = 'human_aware_rl/ppo/data/ppo_runs/2021_07_18-16_39_19_testing_ppo_bc_train_fc_1_random0/seed9456/training_info.pickle'
    # file = open(filename, 'rb')
    # ppo_training_info = pkl.load(file)
    # file.close()
    # print('ppo_training_info = ', ppo_training_info)

    filename = 'human_aware_rl/data/ppo_runs/ppo_bc_models_performance.pickle'
    file = open(filename, 'rb')
    ppo_bc_models_performance = pkl.load(file)
    file.close()
    print('ppo_bc_models_performance = ', ppo_bc_models_performance)
    print()

    filename = 'human_aware_rl/data/ppo_runs/ppo_hm_models_performance.pickle'
    file = open(filename, 'rb')
    ppo_hm_models_performance = pkl.load(file)
    file.close()
    print('ppo_hm_models_performance = ', ppo_hm_models_performance)
    print()

    filename = 'human_aware_rl/data/ppo_runs/ppo_sp_models_performance.pickle'
    file = open(filename, 'rb')
    ppo_sp_models_performance = pkl.load(file)
    file.close()
    print('ppo_sp_models_performance = ', ppo_sp_models_performance)
    print()







