import numpy as np
import pickle as pkl



if __name__ == "__main__":
    # data = {
    #     'PPO_BC_train+BC_test_0': [47.0, 57.0, 60.5, 64.5, 65.5],
    #     'PPO_BC_train+BC_test_1': [82.0, 73.5, 70.0, 75.5, 83.5],
    #     'PPO_BC_test+BC_test_0': [86.5, 76.5, 97.0, 93.0, 101.5],
    #     'PPO_BC_test+BC_test_1': [72.5, 83.0, 78.5, 78.5, 86.5]}
#
#     data = {
#
#     'PPO_BC_train+PPO_BC_train': [77.6, 78.8, 8.0, 60.4, 41.6],
# 'PPO_BC_train+BC_test_0': [53.2, 60.8, 33.4, 50.6, 66.4],
# 'PPO_BC_train+BC_test_1': [72.6, 89.8, 80.4, 87.4, 79.6]
#     }
    data = {

        'PPO_BC_train+BC_test_0': [76.6, 73.0, 69.2, 80.8, 61.0], 'PPO_BC_train+BC_test_1': [99.8, 106.8, 101.6, 77.2, 107.4], 'PPO_BC_test+BC_test_0': [40.4, 44.8, 60.0, 56.8, 63.4], 'PPO_BC_test+BC_test_1': [48.0, 47.4, 60.0, 31.6, 48.4]
        }


    for keyname in data:
        d = data[keyname]
        print(f'key: {keyname}, mean reward = {np.mean(d)}, std = {np.std(d)}')







