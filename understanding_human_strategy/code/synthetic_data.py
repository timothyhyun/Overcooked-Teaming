import random

import numpy as np
from dependencies import *
from hmm import supervised_HMM, unsupervised_HMM, HiddenMarkovModel
from extract_features import *
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import sklearn



def generate_synthetic_data(n_states=5, n_obs=5, sample_emission_length=10, n_iters=20):
    # Compute L and D.
    L = n_states
    D = n_obs

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    # O = [[random.random() for i in range(D)] for j in range(L)]
    O = [[np.random.normal(0, 1) for i in range(D)] for j in range(L)]

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
    A = [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]]

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
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    ss = sklearn.metrics.silhouette_score(X, cluster_labels)
    # ss = sklearn.metrics.calinski_harabasz_score(X, cluster_labels)
    return cluster_labels, cluster_centers, ss


if __name__ == '__main__':
    n_sources = 2
    output_emissions_dict = {}
    for index in range(n_sources):
        output_emissions_dict[index] = {}
        if index % 2 == 0:
            syn_A, syn_O, output_hidden, output_emissions = generate_synthetic_data_manual()
        else:
            syn_A, syn_O, output_hidden, output_emissions = generate_synthetic_data()
        output_emissions_dict[index]['A'] = syn_A
        output_emissions_dict[index]['O'] = syn_O
        output_emissions_dict[index]['hidden'] = output_hidden
        output_emissions_dict[index]['emission'] = output_emissions


    aggregate_emission_data, aggregate_emission_labels = combine_groups_get_labels(output_emissions_dict)
    print('\naggregate_emission_labels = ', aggregate_emission_labels)

    strategy_recog_HMM, hidden_seqs = run_hmm_on_synthetic(aggregate_emission_data, aggregate_emission_labels, n_states=2, window=4, ss=2)

    cluster_labels, cluster_centers, ss = cluster_hidden_states(hidden_seqs, n_clusters=2)
    print('\nhidden_seqs', hidden_seqs)
    print('\ncluster_labels = ', cluster_labels)

    accuracy = 0
    for i in range(len(aggregate_emission_labels)):
        if aggregate_emission_labels[i] == cluster_labels[i]:
            accuracy += 1

    print('acc = ', accuracy/len(aggregate_emission_labels))














