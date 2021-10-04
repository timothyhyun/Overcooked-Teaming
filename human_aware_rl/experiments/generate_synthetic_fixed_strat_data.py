import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import defaultdict

from overcooked_ai_py.utils import save_pickle, load_pickle
from overcooked_ai_py.agents.agent import AgentPair, RandomAgent, CoupledPlanningAgent
from overcooked_ai_py.agents.fixed_strategy_agent import DualPotAgent, FixedStrategy_AgentPair, SinglePotAgent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.utils import save_pickle
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, PlayerState, ObjectState, OvercookedState

import pickle as pkl
import tqdm, copy
import random, os
import pandas as pd
from tqdm import trange
from collections import defaultdict
import ast
import json

from matplotlib.widgets import Slider  # import the Slider widget
from math import pi
from matplotlib.patches import Rectangle, Arrow, FancyArrow

import matplotlib.cm as cm
import cv2
import os
import os
import moviepy.video.io.ImageSequenceClip

# Use Agg backend for canvas
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from understanding_human_strategy.code.dependencies import *
from understanding_human_strategy.code.hmm import supervised_HMM, unsupervised_HMM, HiddenMarkovModel
from understanding_human_strategy.code.extract_features import *


def run_one_game_fixed_agents(a0_type, a1_type):
    layout_name = 'random0'

    simple_mdp = OvercookedGridworld.from_layout_name('random0', start_order_list=['any'], cook_time=20)
    base_params_start_or = {
        'start_orientations': True,
        'wait_allowed': False,
        'counter_goals': simple_mdp.terrain_pos_dict['X'],
        'counter_drop': [],
        'counter_pickup': [],
        'same_motion_goals': False
    }
    mlp = MediumLevelPlanner(simple_mdp, base_params_start_or)
    # random_agent_0 = CoupledPlanningAgent(mlp)
    # random_agent_1 = CoupledPlanningAgent(mlp)

    random_agent_0 = DualPotAgent(simple_mdp, player_index=0)
    random_agent_1 = DualPotAgent(simple_mdp, player_index=1)
    if a0_type == 'SP':
        random_agent_0 = SinglePotAgent(simple_mdp, player_index=0)
    if a1_type == 'SP':
        random_agent_1 = SinglePotAgent(simple_mdp, player_index=0)



    num_rounds = 1
    display = False
    bc_params = {
                'data_params': {
                    'train_mdps': ['random0'],
                    'ordered_trajs': True,
                    'human_ai_trajs': False,
                    'data_path': '../data/human/anonymized/clean_train_trials.pkl'
                },
                 'mdp_params': {
                     'layout_name': 'random0',
                     'start_order_list': None},
                 'env_params': {'horizon': 400},
                 'mdp_fn_params': {}}

    evaluator = AgentEvaluator(mdp_params=bc_params["mdp_params"], env_params=bc_params["env_params"])

    random_results = evaluator.evaluate_agent_pair(
        FixedStrategy_AgentPair(random_agent_0, random_agent_1, allow_duplicate_agents=False),
        num_games=max(int(num_rounds / 2), 1), display=display)


    avg_random_results = np.mean(random_results['ep_returns'])

    # print('avg', avg_random_results)
    return random_results, avg_random_results

def generate_synthetic_fixed(a0_type='DP', a1_type='DP', N_teams=6):
    team_id_to_state_action_sequences = {}
    for i in range(N_teams):
        fixed_results, avg_fixed_results = run_one_game_fixed_agents(a0_type, a1_type)

        action_sequence = fixed_results['ep_actions']
        state_sequence = fixed_results['ep_observations']
        reward_sequence = fixed_results['ep_rewards']

        team_id_to_state_action_sequences[i] = {}
        team_id_to_state_action_sequences[i]['states'] = state_sequence[0]
        team_id_to_state_action_sequences[i]['actions'] = action_sequence[0]
        team_id_to_state_action_sequences[i]['rewards'] = reward_sequence[0]
        team_id_to_state_action_sequences[i]['true_class'] = 0
        team_id_to_state_action_sequences[i]['score'] = avg_fixed_results
        team_id_to_state_action_sequences[i]['time_elapsed'] = list(range(400))

    return team_id_to_state_action_sequences



def plot_trial(p1_data, p2_data, objects_data, title):
    N_steps = len(p1_data)
    a_min = 1  # the minimial value of the paramater a
    a_max = N_steps - 1  # the maximal value of the paramater a
    a_init = 1  # the value of the parameter a to be used initially, when the graph is created

    t = np.linspace(0, N_steps - 1, N_steps)

    layout_shape = 5
    layout = np.array([['X', 'X', 'X', 'P', 'X'],
       ['O', ' ', 'X', '1', 'P'],
       ['O', '2', 'X', ' ', 'X'],
       ['D', ' ', 'X', ' ', 'X'],
       ['X', 'X', 'X', 'S', 'X']], dtype='<U1')

    grid_display = np.zeros((layout_shape, layout_shape, 3))

    for i in range(layout.shape[0]):
        for j in range(layout.shape[1]):
            # Floor = gray
            grid_display[i, j, :] = [220, 220, 220]
            if layout[i, j] == 'X':
                # Counter = Tan
                grid_display[i, j, :] = [91, 153, 91]
            if layout[i, j] == 'P':
                # Pots = brown
                grid_display[i, j, :] = [139, 69, 19]
            if layout[i, j] == 'S':
                # Serve = Green
                grid_display[i, j, :] = [34, 139, 34]
            if layout[i, j] == 'O':
                # Onion = Yellow
                grid_display[i, j, :] = [218, 165, 32]
            if layout[i, j] == 'D':
                # Dishes = Blue
                grid_display[i, j, :] = [65, 105, 225]
            if layout[i, j] == 'T':
                # Tomato = Blue
                grid_display[i, j, :] = [255, 69, 0]

    # loop over your images
    for a in range(len(t)):

        fig = plt.figure(figsize=(8, 3))

        sin_ax = plt.axes([0.1, 0.2, 0.8, 0.65])

        plt.axes(sin_ax)  # select sin_ax

        plt.imshow(grid_display.astype(np.uint8), vmin=0, vmax=255)

        scat1 = plt.scatter(f_p1(t, a, p1_data)[0], f_p1(t, a, p1_data)[1], lw=20, c='r')
        line1, = plt.plot(f_p1(t, a, p1_data)[0], f_p1(t, a, p1_data)[1], lw=5, c='r')

        arrow1 = plt.arrow(arrow_p1(t, a, p1_data)[0], arrow_p1(t, a, p1_data)[1],
                           arrow_p1(t, a, p1_data)[2], arrow_p1(t, a, p1_data)[3], head_width=0.5,
                           head_length=0.5, width=0.02, fc='r', ec='r', length_includes_head=True)

        if obj_p1(t, a, p1_data)[0] is not None:
            arrow_obj1 = plt.scatter(obj_p1(t, a, p1_data)[0], obj_p1(t, a, p1_data)[1],
                                     lw=10, c=obj_p1(t, a, p1_data)[3])

        scat2 = plt.scatter(f_p2(t, a, p2_data)[0], f_p2(t, a, p2_data)[1], lw=20, c='b')
        line2, = plt.plot(f_p2(t, a, p2_data)[0], f_p2(t, a, p2_data)[1], lw=5, c='b')

        arrow2 = plt.arrow(arrow_p2(t, a, p2_data)[0], arrow_p2(t, a, p2_data)[1],
                           arrow_p2(t, a, p2_data)[2], arrow_p2(t, a, p2_data)[3], head_width=0.5,
                           head_length=0.5, width=0.02, fc='b', ec='b', length_includes_head=True)

        if obj_p2(t, a, p2_data)[0] is not None:
            arrow_obj2 = plt.scatter(obj_p2(t, a, p2_data)[0], obj_p2(t, a, p2_data)[1],
                                     lw=10, c=obj_p2(t, a, p2_data)[3])

        objects_list = world_obj(t, a, objects_data)
        for obj in objects_list:
            # print('obj', obj)
            obj_add = plt.scatter(obj[0], obj[1],
                                  lw=10, c=obj[3])
        plt.title(title)
        plt.savefig('imgs_test/im_' + str(a) + '.png')
        # plt.savefig('imgs_test/im_' + str(a) + '_1.png')

        plt.close()



def visualize_trial(a0_type, a1_type, title="FC: DP vs. DP Players", video_filename='my_video'):

    fixed_results, avg_fixed_results = run_one_game_fixed_agents(a0_type, a1_type)

    joint_actions = fixed_results['ep_actions'][0]
    state_data = fixed_results['ep_observations'][0]
    time_elapsed = list(range(400))
    score = avg_fixed_results

    p1_data = []
    p2_data = []
    p1_actions = []
    p2_actions = []
    state_data_eval = []
    objects_data = []
    for i in range(1, len(state_data)):
        prev_state_x = state_data[i-1].to_dict()
        state_x = state_data[i].to_dict()
        joint_actions_i = joint_actions[i]
        p1_index = 1
        p2_index = 0

        p1_data.append(state_x['players'][p1_index])
        p2_data.append(state_x['players'][p2_index])
        state_data_eval.append(state_x)
        objects_data.append(state_x['objects'])

        p1_actions.append(joint_actions_i[p1_index])
        p2_actions.append(joint_actions_i[p2_index])

    print("PLOTTING TRIAL...................")
    plot_trial(p1_data, p2_data, objects_data, title)
    print("SAVE IMAGES...................")
    save_images_to_video(video_filename)

    print("CLEARING DIRECTORY...................")
    clear_images_directory()


def clear_images_directory():
    dir_name = "imgs_test"
    test = os.listdir(dir_name)

    for item in test:
        if item.endswith(".png"):
            os.remove(os.path.join(dir_name, item))

def save_images_to_video(videofile):
    image_folder = 'imgs_test'
    fps = 1

    image_files = sorted([image_folder + '/' + img for img in os.listdir(image_folder) if img.endswith(".png")])
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(f'imgs_test/{videofile}.mp4')

if __name__ == "__main__":
    visualize_trial('DP', 'DP', title="FC: DP vs. DP Players", video_filename='dp_dp_video_2')
    visualize_trial('SP', 'SP', title="FC: SP vs. SP Players", video_filename='sp_sp_video_2')
    visualize_trial('DP', 'SP', title="FC: SP vs. DP Players", video_filename='dp_sp_video_2')
    visualize_trial('SP', 'DP', title="FC: DP vs. SP Players", video_filename='sp_dp_video_2')







