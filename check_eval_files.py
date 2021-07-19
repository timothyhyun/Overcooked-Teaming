import numpy as np
import pickle as pkl



if __name__ == "__main__":
    filename = 'human_aware_rl/data/bc_runs/best_bc_models_performance.pickle'
    file = open(filename, 'rb')
    best_models_perf = pkl.load(file)
    file.close()
    print('best_models_perf = ', best_models_perf)

    filename = 'human_aware_rl/data/bc_runs/bc_models_all_evaluations.pickle'
    file = open(filename, 'rb')
    best_models_evals = pkl.load(file)
    file.close()
    print('best_models_evals = ', best_models_evals)







