import random

import matplotlib.pyplot as plt
import numpy as np
from dependencies import *
from hmm import supervised_HMM, unsupervised_HMM, HiddenMarkovModel
from extract_features import *
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import sklearn
import pickle
from cluster_leven import perform_clustering_on_int_strings

# from leven import levenshtein
#
# def lev_metric(x, y):
#     i, j = int(x[0]), int(y[0])     # extract indices
#     return levenshtein(data[i], data[j])

def generate_synthetic_data(A_mean, A_std, O_mean, O_std, n_states=5, n_obs=5, sample_emission_length=10, n_iters=20):
    # Compute L and D.
    L = n_states
    D = n_obs

    # Randomly initialize and normalize matrix A.
    A = [[np.random.normal(A_mean, A_std) for i in range(L)] for j in range(L)]
    A = [[(-A[i][j] if A[i][j] < 0 else A[i][j]) for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    # O = [[random.random() for i in range(D)] for j in range(L)]
    O = [[np.random.normal(O_mean, O_std) for i in range(D)] for j in range(L)]
    O = [[(-O[i][j] if O[i][j] < 0 else O[i][j]) for i in range(L)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm


    syn_A = A
    syn_O = O

    syn_hmm = HiddenMarkovModel(syn_A, syn_O)

    output_hidden = []
    output_emissions = []
    for iteration in range(n_iters):
        sample_emission, sample_hidden_states = syn_hmm.generate_emission(sample_emission_length)
        output_hidden.append(sample_hidden_states)
        output_emissions.append(sample_emission)

    # output_hidden = np.array(output_hidden)
    # output_emissions = np.array(output_emissions)
    return syn_A, syn_O, output_hidden, output_emissions


def generate_synthetic_noiseless_manual_data(A, O, n_states, n_obs, sample_emission_length=10, n_iters=20):

    # Compute L and D.
    L = n_states
    D = n_obs

    # Randomly initialize and normalize matrix A.

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    # O = [[random.random() for i in range(D)] for j in range(L)]


    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm


    syn_A = A
    syn_O = O

    syn_hmm = HiddenMarkovModel(syn_A, syn_O)

    output_hidden = []
    output_emissions = []
    for iteration in range(n_iters):
        sample_emission, sample_hidden_states = syn_hmm.generate_emission(sample_emission_length)
        output_hidden.append(sample_hidden_states)
        output_emissions.append(sample_emission)

    # output_hidden = np.array(output_hidden)
    # output_emissions = np.array(output_emissions)
    return syn_A, syn_O, output_hidden, output_emissions


def generate_synthetic_data_manual(n_states=5, n_obs=5, sample_emission_length=10, n_iters=20):
    # Compute L and D.
    L = n_states
    D = n_obs

    # Randomly initialize and normalize matrix A.


    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[np.random.normal(3,1) for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm


    syn_A = A
    syn_O = O

    syn_hmm = HiddenMarkovModel(syn_A, syn_O)

    output_hidden = []
    output_emissions = []
    for iteration in range(n_iters):
        sample_emission, sample_hidden_states = syn_hmm.generate_emission(sample_emission_length)
        output_hidden.append(sample_hidden_states)
        output_emissions.append(sample_emission)

    # output_hidden = np.array(output_hidden)
    # output_emissions = np.array(output_emissions)
    return syn_A, syn_O, output_hidden, output_emissions


def combine_groups_get_labels(output_emissions_dict):
    aggregate_emission_data = []
    aggregate_emission_labels = []
    for emission_idx in output_emissions_dict:
        for item in output_emissions_dict[emission_idx]['emission']:
            aggregate_emission_data.append(np.array(item))
            aggregate_emission_labels.append(emission_idx)

    return aggregate_emission_data, aggregate_emission_labels


def run_hmm_on_synthetic(aggregate_emission_data, aggregate_emission_labels, n_states=5, window=4, ss=2):

    # X = observation_data
    # Y = hidden_state_data
    # X, team_numbers, X_data_chunked, team_numbers_chunked = featurize_data_for_chunked_naive_hmm(window=window, ss=ss)


    N_iters = 100
    X = aggregate_emission_data
    strategy_recog_HMM = unsupervised_HMM(X, n_states, N_iters)

    # print('emission', test_unsuper_hmm.generate_emission(10))
    hidden_seqs = []
    # team_num_to_seq_probs = {}
    for j in range(len(X)):
        viterbi_output, all_sequences_and_probs = strategy_recog_HMM.viterbi_all_probs(X[j])
        # team_num_to_seq_probs[team_numbers[j]] = all_sequences_and_probs
        hidden_seqs.append([int(x) for x in viterbi_output])
        # print('viterbi: hidden seq: Team ' + str(team_numbers[j]) + ": ", viterbi_output)

    return strategy_recog_HMM, hidden_seqs


def cluster_hidden_states(hidden_seqs, n_clusters=2):
    X = np.array(hidden_seqs)
    # print('X=', X)
    ## METHOD 1: Euclidean
    # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    # cluster_labels = kmeans.labels_
    # cluster_centers = kmeans.cluster_centers_

    ## METHOD 2: Leven, Edit Distance
    cluster_labels, cluster_centers = perform_clustering_on_int_strings(X, n_clusters)
    # print('cluster_labels', cluster_labels)
    # print('cluster_centers', cluster_centers)

    ss = sklearn.metrics.silhouette_score(X, cluster_labels)
    # ss = sklearn.metrics.calinski_harabasz_score(X, cluster_labels)
    return cluster_labels, cluster_centers, ss

def get_best_hmm(aggregate_emission_data, aggregate_emission_labels):
    # candidate_N_hidden = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # # candidate_K_clusters = [2, 3, 4, 5]
    # candidate_K_clusters = [2, 3, 4]
    candidate_N_hidden = [4]
    candidate_K_clusters = [2]

    min_ss = -10000
    best_n_hidden = 0
    best_k_clusters = 0
    best_cluster_labels = None

    for n_hidden_states in candidate_N_hidden:
        for k_clusters in candidate_K_clusters:
            strategy_recog_HMM, hidden_seqs = run_hmm_on_synthetic(aggregate_emission_data, aggregate_emission_labels,
                                                                   n_states=n_hidden_states, window=4, ss=2)

            cluster_labels, cluster_centers, ss = cluster_hidden_states(hidden_seqs, n_clusters=k_clusters)
            if ss > min_ss:
                min_ss = ss
                best_n_hidden = n_hidden_states
                best_k_clusters = k_clusters
                best_cluster_labels = cluster_labels


    return min_ss, best_n_hidden, best_k_clusters, best_cluster_labels


def get_accuracy(aggregate_emission_labels, best_cluster_labels, best_k_clusters, n_sources):
    n_clusters_correct = True if best_k_clusters == n_sources else False
    if n_clusters_correct:
        accuracy = 0
        for i in range(len(aggregate_emission_labels)):
            if aggregate_emission_labels[i] == best_cluster_labels[i]:
                accuracy += 1

        acc = accuracy / len(aggregate_emission_labels)
        if acc < 0.5:
            acc = 1 - acc
        return acc

    else:
        # accuracy = 0
        avg_acc_for_all_clusters = {}
        # pred_cluster_specific_labels = {}
        true_cluster_specific_labels = {}
        for k in range(best_k_clusters):
            # pred_cluster_specific_labels[k] = []
            true_cluster_specific_labels[k] = []
            avg_acc_for_all_clusters[k] = []
        for i in range(len(aggregate_emission_labels)):
            c_label = aggregate_emission_labels[i]
            true_cluster_specific_labels[c_label].append(aggregate_emission_labels[i])

        avg_purity_score_per_cluster = []
        for k in range(best_k_clusters):
            n_datapoints_in_cluster = len(true_cluster_specific_labels[k])

            labels_in_cluster = true_cluster_specific_labels[k]
            source_class_counts = []
            for source_class in range(n_sources):
                class_count = np.count_nonzero(labels_in_cluster == source_class)
                source_class_counts.append(class_count)

            n_in_max_class = np.max(source_class_counts)
            purity_of_max_class = n_in_max_class/n_datapoints_in_cluster
            avg_purity_score_per_cluster.append(purity_of_max_class)

        avg_purity = np.mean(avg_purity_score_per_cluster)
        return avg_purity


def get_acc_of_config(A_mean, A_std, O_mean, O_std, N_hidden_states, n_sources=2):

    output_emissions_dict = {}
    for index in range(n_sources):
        output_emissions_dict[index] = {}
        syn_A, syn_O, output_hidden, output_emissions = generate_synthetic_data(A_mean, A_std, O_mean, O_std)
        output_emissions_dict[index]['A'] = syn_A
        output_emissions_dict[index]['O'] = syn_O
        output_emissions_dict[index]['hidden'] = output_hidden
        output_emissions_dict[index]['emission'] = output_emissions

    A_distance = []
    O_distance = []
    A_1 = output_emissions_dict[0]['A']
    O_1 = output_emissions_dict[0]['O']
    A_2 = output_emissions_dict[1]['A']
    O_2 = output_emissions_dict[1]['O']
    for i in range(len(A_1)):
        for j in range(len(A_1[0])):
            A_distance.append(abs(A_1[i][j] - A_2[i][j]))

    for i in range(len(O_1)):
        for j in range(len(O_1[0])):
            O_distance.append(abs(O_1[i][j] - O_2[i][j]))

    A_distance = np.mean(A_distance)
    O_distance = np.mean(O_distance)
    AO_distance = np.mean([A_distance, O_distance])

    aggregate_emission_data, aggregate_emission_labels = combine_groups_get_labels(output_emissions_dict)
    # print('\naggregate_emission_labels = ', aggregate_emission_labels)

    min_ss, best_n_hidden, best_k_clusters, best_cluster_labels = get_best_hmm(aggregate_emission_data, aggregate_emission_labels)
    # print('\nhidden_seqs', hidden_seqs)
    # print('\ncluster_labels = ', cluster_labels)

    acc = get_accuracy(aggregate_emission_labels, best_cluster_labels, best_k_clusters, n_sources)



    print(f'A_mean={A_mean}, A_std={A_std}, O_mean={O_mean}, O_std={O_std}, A_distance={A_distance}, O_distance={O_distance}, ss={min_ss}: acc = {acc}')
    return acc, A_distance, O_distance, AO_distance, min_ss


def manual_experiment(A1, O1, A2, O2, N_hidden_states, n_states, n_obs):
    n_sources = 2
    output_emissions_dict = {}

    index = 0
    output_emissions_dict[index] = {}
    O = O1
    A = A1
    syn_A, syn_O, output_hidden, output_emissions = generate_synthetic_noiseless_manual_data(A, O, n_states=n_states, n_obs=n_obs, sample_emission_length=10, n_iters=20)
    output_emissions_dict[index]['A'] = syn_A
    output_emissions_dict[index]['O'] = syn_O
    output_emissions_dict[index]['hidden'] = output_hidden
    output_emissions_dict[index]['emission'] = output_emissions

    index = 1
    output_emissions_dict[index] = {}
    A = A2
    O = O2
    syn_A, syn_O, output_hidden, output_emissions = generate_synthetic_noiseless_manual_data(A, O, n_states=4, n_obs=n_obs,
                                                                                             sample_emission_length=10,
                                                                                             n_iters=20)
    output_emissions_dict[index]['A'] = syn_A
    output_emissions_dict[index]['O'] = syn_O
    output_emissions_dict[index]['hidden'] = output_hidden
    output_emissions_dict[index]['emission'] = output_emissions

    A_distance = []
    O_distance = []
    A_1 = output_emissions_dict[0]['A']
    O_1 = output_emissions_dict[0]['O']
    A_2 = output_emissions_dict[1]['A']
    O_2 = output_emissions_dict[1]['O']
    for i in range(len(A_1)):
        for j in range(len(A_1[0])):
            A_distance.append(abs(A_1[i][j] - A_2[i][j]))

    for i in range(len(O_1)):
        for j in range(len(O_1[0])):
            O_distance.append(abs(O_1[i][j] - O_2[i][j]))

    A_distance = np.mean(A_distance)
    O_distance = np.mean(O_distance)

    aggregate_emission_data, aggregate_emission_labels = combine_groups_get_labels(output_emissions_dict)
    print('\naggregate_emission_labels = ', aggregate_emission_labels)

    strategy_recog_HMM, hidden_seqs = run_hmm_on_synthetic(aggregate_emission_data, aggregate_emission_labels,
                                                           n_states=N_hidden_states, window=4, ss=2)

    cluster_labels, cluster_centers, ss = cluster_hidden_states(hidden_seqs, n_clusters=2)
    # print('\nhidden_seqs', hidden_seqs)
    print('\ncluster_labels = ', cluster_labels)

    accuracy = 0
    for i in range(len(aggregate_emission_labels)):
        if aggregate_emission_labels[i] == cluster_labels[i]:
            accuracy += 1

    acc = accuracy / len(aggregate_emission_labels)


    if acc < 0.5:
        acc = 1-acc
    print(f'A_distance={A_distance}, O_distance={O_distance}: acc = {acc}')
    return acc, A_distance, O_distance

def run_experiment(N_hidden_states):

    # Define a set of A_means, A_stds
    all_A_means = [0, 3, 5, 10]
    all_A_stds = [0.1, 1, 5, 10]

    # Define a set of O_means, O_stds
    all_O_means = [0, 1, 3, 5, 10]
    all_O_stds = [0.1, 1, 5, 10]

    output_dict = {}

    all_A_distances = []
    all_O_distances = []
    all_AO_distances = []
    all_accuracies = []

    all_A_mean_list = []
    all_A_std_list = []
    all_O_mean_list = []
    all_O_std_list = []

    all_ss_list = []

    for Am_i in range(len(all_A_means)):
        A_mean = all_A_means[Am_i]
        for As_i in range(len(all_A_stds)):
            A_std = all_A_stds[As_i]
            for Om_i in range(len(all_O_means)):
                O_mean = all_O_means[Om_i]
                for Os_i in range(len(all_O_stds)):
                    O_std = all_O_stds[Os_i]


                    acc, A_distance, O_distance, AO_distance, ss = get_acc_of_config(A_mean, A_std, O_mean, O_std, N_hidden_states)
                    output_dict[(A_mean, A_std, O_mean, O_std, A_distance, O_distance, ss)] = acc

                    all_A_distances.append(A_distance)
                    all_O_distances.append(O_distance)
                    all_accuracies.append(acc)
                    all_A_mean_list.append(A_mean)
                    all_A_std_list.append(A_std)
                    all_O_mean_list.append(O_mean)
                    all_O_std_list.append(O_std)
                    all_ss_list.append(ss)
                    all_AO_distances.append(AO_distance)

    test_number = 9

    filehandler = open(f"output_dict_{test_number}.pkl", "wb")
    pickle.dump(output_dict, filehandler)
    filehandler.close()


    N_hidden_states = 'N'
    plt.figure()
    plt.scatter(all_A_distances, all_accuracies)
    plt.xlabel("Transition Matrix Distance: A Matrix")
    plt.ylabel("Classification Accuracy")
    plt.title(f"{N_hidden_states} Hidden States, 2 Groups: A distance vs. Accuracy")
    plt.savefig(f"test{test_number}_Adist_vs_Acc.png")
    plt.close()

    plt.figure()
    plt.scatter(all_O_distances, all_accuracies)
    plt.xlabel("Emission Matrix Distance: O Matrix")
    plt.ylabel("Classification Accuracy")
    plt.title(f"{N_hidden_states} Hidden States, 2 Groups: O distance vs. Accuracy")
    plt.savefig(f"test{test_number}_Odist_vs_Acc.png")
    plt.close()

    plt.figure()
    plt.scatter(all_AO_distances, all_accuracies)
    plt.xlabel("Avg Distance over Transition and Emission Matrix")
    plt.ylabel("Classification Accuracy")
    plt.title(f"{N_hidden_states} Hidden States, 2 Groups: AO distance vs. Accuracy")
    plt.savefig(f"test{test_number}_AOdist_vs_Acc.png")
    plt.close()

    plt.figure()
    plt.scatter(all_A_mean_list, all_accuracies)
    plt.xlabel("Transition Matrix Mean")
    plt.ylabel("Classification Accuracy")
    plt.title(f"{N_hidden_states} Hidden States, 2 Groups: A Mean vs. Accuracy")
    plt.savefig(f"test{test_number}_Amean_vs_Acc.png")
    plt.close()

    plt.figure()
    plt.scatter(all_A_std_list, all_accuracies)
    plt.xlabel("Transition Matrix Std")
    plt.ylabel("Classification Accuracy")
    plt.title(f"{N_hidden_states} Hidden States, 2 Groups: A Std vs. Accuracy")
    plt.savefig(f"test{test_number}_Astd_vs_Acc.png")
    plt.close()

    plt.figure()
    plt.scatter(all_O_mean_list, all_accuracies)
    plt.xlabel("Emission Matrix Mean")
    plt.ylabel("Classification Accuracy")
    plt.title(f"{N_hidden_states} Hidden States, 2 Groups: O Mean vs. Accuracy")
    plt.savefig(f"test{test_number}_Omean_vs_Acc.png")
    plt.close()

    plt.figure()
    plt.scatter(all_O_std_list, all_accuracies)
    plt.xlabel("Emission Matrix Std")
    plt.ylabel("Classification Accuracy")
    plt.title(f"{N_hidden_states} Hidden States, 2 Groups: O Std vs. Accuracy")
    plt.savefig(f"test{test_number}_Ostd_vs_Acc.png")
    plt.close()

    plt.figure()
    plt.scatter(all_ss_list, all_accuracies)
    plt.xlabel("Silhouette Score")
    plt.ylabel("Classification Accuracy")
    plt.title(f"{N_hidden_states} Hidden States, 2 Groups: SScore vs. Accuracy")
    plt.savefig(f"test{test_number}_SS_vs_Acc.png")
    plt.close()




if __name__ == '__main__':
    # N_hidden_states=4
    # run_experiment(N_hidden_states)
    import scipy
    # import cPickle as pickle
    test_number = 3
    with open("output_dict_4.pkl", "rb") as filename:
        output_dict = pickle.load(filename)

    x = []
    y = []
    for keyname in output_dict:
        acc = output_dict[keyname]
        (A_mean, A_std, O_mean, O_std, A_distance, O_distance, ss) = keyname
        x.append(A_distance)
        y.append(acc)

    r, p = scipy.stats.pearsonr(x, y)
    print('r = ', r)
    print('p = ', p)

    # EXPERIMENT 1
    # print("RUNNING EXPERIMENT 1.........")
    # O1 = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]
    # A1 = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    #
    # A2 = [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]
    # O2 = [[0, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0]]
    #
    # N_hidden_states = 4
    # n_states = 3
    # n_obs = 5
    #
    # manual_experiment(A1, O1, A2, O2, N_hidden_states, n_states, n_obs)

    # EXPERIMENT 2
    # print("RUNNING EXPERIMENT 2.........")
    # O1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # A1 = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    #
    # # A2 = [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]
    #
    # O2 = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    # A2 = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    #
    # N_hidden_states = 9
    # n_states = 3
    # n_obs = 5
    # manual_experiment(A1, O1, A2, O2, N_hidden_states,  n_states, n_obs)
    #
    # # EXPERIMENT 3
    # print("RUNNING EXPERIMENT 3.........")
    # O1 = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]
    # A1 = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    #
    # O2 = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]
    # A2 = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
    #
    # N_hidden_states = 8
    # n_states = 3
    # n_obs = 5
    #
    # manual_experiment(A1, O1, A2, O2, N_hidden_states, n_states, n_obs)











