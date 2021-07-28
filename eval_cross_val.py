
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt



if __name__ == "__main__":


    cval_results =  {0: {'subset': (19, 3, 11, 13), 'train_workers': [19, 3, 11, 13], 'test_workers': [2, 4, 17, 1, 10, 12], 'best_bc_models_performance': {'random0': {'BC_train+BC_train': (16.0, 5.513619500836089), 'BC_test+BC_test': (36.0, 10.881176406988354), 'BC_train+BC_test_0': (36.0, 10.507140429250956), 'BC_train+BC_test_1': (22.0, 5.966573556070518)}}}, 1: {'subset': (19, 10, 11, 13), 'train_workers': [19, 10, 11, 13], 'test_workers': [2, 4, 17, 1, 3, 12], 'best_bc_models_performance': {'random0': {'BC_train+BC_train': (38.0, 8.694826047713663), 'BC_test+BC_test': (34.0, 8.508818954473059), 'BC_train+BC_test_0': (28.0, 9.465727652959385), 'BC_train+BC_test_1': (24.0, 7.37563556583431)}}}, 2: {'subset': (17, 10, 12, 13), 'train_workers': [17, 10, 12, 13], 'test_workers': [2, 4, 19, 1, 3, 11], 'best_bc_models_performance': {'random0': {'BC_train+BC_train': (30.0, 9.899494936611665), 'BC_test+BC_test': (32.0, 9.465727652959385), 'BC_train+BC_test_0': (36.0, 12.263767773404714), 'BC_train+BC_test_1': (38.0, 9.57078889120432)}}}, 3: {'subset': (2, 19, 3, 13), 'train_workers': [2, 19, 3, 13], 'test_workers': [4, 17, 1, 10, 11, 12], 'best_bc_models_performance': {'random0': {'BC_train+BC_train': (30.0, 8.12403840463596), 'BC_test+BC_test': (18.0, 9.979979959899719), 'BC_train+BC_test_0': (44.0, 10.119288512538814), 'BC_train+BC_test_1': (30.0, 9.899494936611665)}}}, 4: {'subset': (2, 17, 11, 13), 'train_workers': [2, 17, 11, 13], 'test_workers': [4, 19, 1, 3, 10, 12], 'best_bc_models_performance': {'random0': {'BC_train+BC_train': (24.0, 6.196773353931866), 'BC_test+BC_test': (32.0, 5.059644256269407), 'BC_train+BC_test_0': (40.0, 9.797958971132712), 'BC_train+BC_test_1': (42.0, 12.47397290361014)}}}, 5: {'subset': (2, 4, 17, 12), 'train_workers': [2, 4, 17, 12], 'test_workers': [19, 1, 3, 10, 11, 13], 'best_bc_models_performance': {'random0': {'BC_train+BC_train': (14.0, 5.692099788303082), 'BC_test+BC_test': (32.0, 10.276186062932103), 'BC_train+BC_test_0': (34.0, 8.966604708583958), 'BC_train+BC_test_1': (28.0, 10.658330075579382)}}}, 6: {'subset': (19, 3, 11, 13), 'train_workers': [19, 3, 11, 13], 'test_workers': [2, 4, 17, 1, 10, 12], 'best_bc_models_performance': {'random0': {'BC_train+BC_train': (16.0, 5.513619500836089), 'BC_test+BC_test': (36.0, 10.881176406988354), 'BC_train+BC_test_0': (36.0, 10.507140429250956), 'BC_train+BC_test_1': (22.0, 5.966573556070518)}}}, 7: {'subset': (4, 19, 12, 13), 'train_workers': [4, 19, 12, 13], 'test_workers': [2, 17, 1, 3, 10, 11], 'best_bc_models_performance': {'random0': {'BC_train+BC_train': (26.0, 8.024961059095551), 'BC_test+BC_test': (40.0, 8.48528137423857), 'BC_train+BC_test_0': (18.0, 7.720103626247512), 'BC_train+BC_test_1': (30.0, 9.055385138137416)}}}, 8: {'subset': (19, 3, 11, 12), 'train_workers': [19, 3, 11, 12], 'test_workers': [2, 4, 17, 1, 10, 13], 'best_bc_models_performance': {'random0': {'BC_train+BC_train': (18.0, 8.221921916437786), 'BC_test+BC_test': (18.0, 8.694826047713663), 'BC_train+BC_test_0': (30.0, 10.295630140986999), 'BC_train+BC_test_1': (40.0, 8.94427190999916)}}}, 9: {'subset': (17, 19, 3, 11), 'train_workers': [17, 19, 3, 11], 'test_workers': [2, 4, 1, 10, 12, 13], 'best_bc_models_performance': {'random0': {'BC_train+BC_train': (8.0, 3.098386676965933), 'BC_test+BC_test': (16.0, 7.37563556583431), 'BC_train+BC_test_0': (22.0, 8.694826047713663), 'BC_train+BC_test_1': (50.0, 8.12403840463596)}}}, 10: {'subset': (19, 1, 3, 12), 'train_workers': [19, 1, 3, 12], 'test_workers': [2, 4, 17, 10, 11, 13], 'best_bc_models_performance': {'random0': {'BC_train+BC_train': (16.0, 7.899367063252599), 'BC_test+BC_test': (22.0, 8.694826047713663), 'BC_train+BC_test_0': (46.0, 11.679041056525145), 'BC_train+BC_test_1': (36.0, 9.715966241192895)}}}, 11: {'subset': (1, 3, 11, 13), 'train_workers': [1, 3, 11, 13], 'test_workers': [2, 4, 17, 19, 10, 12], 'best_bc_models_performance': {'random0': {'BC_train+BC_train': (30.0, 11.40175425099138), 'BC_test+BC_test': (22.0, 7.720103626247512), 'BC_train+BC_test_0': (46.0, 8.966604708583958), 'BC_train+BC_test_1': (28.0, 8.099382692526634)}}}, 12: {'subset': (2, 17, 11, 12), 'train_workers': [2, 17, 11, 12], 'test_workers': [4, 19, 1, 3, 10, 13], 'best_bc_models_performance': {'random0': {'BC_train+BC_train': (14.0, 5.692099788303082), 'BC_test+BC_test': (34.0, 8.024961059095551), 'BC_train+BC_test_0': (40.0, 6.928203230275509), 'BC_train+BC_test_1': (42.0, 3.40587727318528)}}}, 13: {'subset': (4, 17, 1, 3), 'train_workers': [4, 17, 1, 3], 'test_workers': [2, 19, 10, 11, 12, 13], 'best_bc_models_performance': {'random0': {'BC_train+BC_train': (38.0, 11.117553687749837), 'BC_test+BC_test': (34.0, 5.692099788303082), 'BC_train+BC_test_0': (28.0, 5.796550698475775), 'BC_train+BC_test_1': (28.0, 8.099382692526634)}}}, 14: {'subset': (1, 3, 12, 13), 'train_workers': [1, 3, 12, 13], 'test_workers': [2, 4, 17, 19, 10, 11], 'best_bc_models_performance': {'random0': {'BC_train+BC_train': (24.0, 6.81175454637056), 'BC_test+BC_test': (28.0, 9.879271228182773), 'BC_train+BC_test_0': (34.0, 4.939635614091387), 'BC_train+BC_test_1': (20.0, 6.324555320336758)}}}, 15: {'subset': (4, 17, 3, 13), 'train_workers': [4, 17, 3, 13], 'test_workers': [2, 19, 1, 10, 11, 12], 'best_bc_models_performance': {'random0': {'BC_train+BC_train': (22.0, 7.720103626247512), 'BC_test+BC_test': (26.0, 8.024961059095551), 'BC_train+BC_test_0': (22.0, 5.966573556070518), 'BC_train+BC_test_1': (28.0, 8.579044235810887)}}}, 16: {'subset': (2, 4, 1, 10), 'train_workers': [2, 4, 1, 10], 'test_workers': [17, 19, 3, 11, 12, 13], 'best_bc_models_performance': {'random0': {'BC_train+BC_train': (18.0, 6.603029607687671), 'BC_test+BC_test': (26.0, 12.345039489608771), 'BC_train+BC_test_0': (42.0, 14.818906842274162), 'BC_train+BC_test_1': (18.0, 6.603029607687671)}}}, 17: {'subset': (2, 4, 17, 3), 'train_workers': [2, 4, 17, 3], 'test_workers': [19, 1, 10, 11, 12, 13], 'best_bc_models_performance': {'random0': {'BC_train+BC_train': (22.0, 6.603029607687671), 'BC_test+BC_test': (42.0, 8.694826047713663), 'BC_train+BC_test_0': (38.0, 9.57078889120432), 'BC_train+BC_test_1': (38.0, 8.694826047713663)}}}, 18: {'subset': (17, 19, 1, 10), 'train_workers': [17, 19, 1, 10], 'test_workers': [2, 4, 3, 11, 12, 13], 'best_bc_models_performance': {'random0': {'BC_train+BC_train': (42.0, 7.720103626247512), 'BC_test+BC_test': (48.0, 9.465727652959385), 'BC_train+BC_test_0': (32.0, 8.579044235810887), 'BC_train+BC_test_1': (42.0, 8.694826047713663)}}}, 19: {'subset': (4, 10, 12, 13), 'train_workers': [4, 10, 12, 13], 'test_workers': [2, 17, 19, 1, 3, 11], 'best_bc_models_performance': {'random0': {'BC_train+BC_train': (44.0, 10.507140429250956), 'BC_test+BC_test': (52.0, 12.066482503198683), 'BC_train+BC_test_0': (20.0, 5.65685424949238), 'BC_train+BC_test_1': (14.0, 6.356099432828281)}}}}


    results_means = {}
    results_means['BC_train+BC_train'] = []
    results_means['BC_test+BC_test'] = []
    results_means['BC_train+BC_test_0'] = []
    results_means['BC_train+BC_test_1'] = []

    best_groups = {}
    best_groups['BC_train+BC_train'] = {}
    best_groups['BC_test+BC_test'] =  {}
    best_groups['BC_train+BC_test_0'] =  {}
    best_groups['BC_train+BC_test_1'] =  {}
    for keyname in best_groups:
        best_groups[keyname]['value'] = 0
        best_groups[keyname]['train_workers'] = []
        best_groups[keyname]['test_workers'] = []

    plot_x_vals = []
    for key_val in cval_results:
        plot_x_vals.append(key_val)
        train_set = cval_results[key_val]['train_workers']
        test_set = cval_results[key_val]['test_workers']
        performance = cval_results[key_val]['best_bc_models_performance']['random0']
        for keyname in results_means:
            perf = performance[keyname][0]
            results_means[keyname].append(perf)
            if perf > best_groups[keyname]['value']:
                best_groups[keyname]['value'] = perf
                best_groups[keyname]['train_workers'] = train_set
                best_groups[keyname]['test_workers'] = test_set

    for keyname in results_means:
        plt.scatter(plot_x_vals, results_means[keyname])
        plt.title(keyname)
        plt.xlabel("Index")
        plt.ylabel("Avg Reward (N=10)")
        plt.savefig(f"{keyname}_scatter.png")
        plt.close()

        plt.hist(results_means[keyname])
        plt.title(keyname)
        plt.xlabel("Avg Reward (N=10)")
        plt.ylabel("Count (# combinations)")
        plt.savefig(f"{keyname}_hist.png")
        plt.close()

    print('best_groups = ', best_groups)
    pkl.dump(best_groups, open('cval_results_n10_best_groups.pkl', 'wb'))


"""
best_groups =  {
'BC_train+BC_train': {'value': 44.0, 'train_workers': [4, 10, 12, 13], 'test_workers': [2, 17, 19, 1, 3, 11]}, 
'BC_test+BC_test': {'value': 52.0, 'train_workers': [4, 10, 12, 13], 'test_workers': [2, 17, 19, 1, 3, 11]}, 
'BC_train+BC_test_0': {'value': 46.0, 'train_workers': [19, 1, 3, 12], 'test_workers': [2, 4, 17, 10, 11, 13]}, 
'BC_train+BC_test_1': {'value': 50.0, 'train_workers': [17, 19, 3, 11], 'test_workers': [2, 4, 1, 10, 12, 13]}}

TRUE GRUPS
# Train Worker ID to Team: {2: 14, 4: 24, 15: 79, 17: 89, 19: 99, 22: 114}  = [2, 4, 17, 19], leave 15, 22
# Test Worker ID to Team: {1: 9, 3: 19, 10: 54, 11: 59, 12: 64, 13: 69} = [1, 3, 10, 11, 12, 13]
"""








