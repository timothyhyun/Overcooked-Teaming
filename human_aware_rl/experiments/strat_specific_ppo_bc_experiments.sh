#python ../ppo/ppo.py with EX_NAME="ppo_bc_train_simple" layout_name="simple" REW_SHAPING_HORIZON=1e6 PPO_RUN_TOT_TIMESTEPS=8e6 LR=1e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456, 1887, 5578, 5987,  516]" VF_COEF=0.5 MINIBATCHES=10 LR_ANNEALING=3 SELF_PLAY_HORIZON="[5e5, 3e6]" TIMESTAMP_DIR=False
#python ../ppo/ppo.py with EX_NAME="ppo_bc_test_simple" layout_name="simple" REW_SHAPING_HORIZON=1e6 PPO_RUN_TOT_TIMESTEPS=8e6 LR=1e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_test" SEEDS="[2888, 7424, 7360, 4467,  184]" VF_COEF=0.5 MINIBATCHES=10 LR_ANNEALING=3 SELF_PLAY_HORIZON="[5e5, 3e6]" TIMESTAMP_DIR=False
#
#python ../ppo/ppo.py with EX_NAME="ppo_bc_train_unident_s" layout_name="unident_s" REW_SHAPING_HORIZON=6e6 PPO_RUN_TOT_TIMESTEPS=1e7 LR=1e-3 GPU_ID=1 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456, 1887, 5578, 5987,  516]" VF_COEF=0.5 MINIBATCHES=12 LR_ANNEALING=3 SELF_PLAY_HORIZON="[1e6, 7e6]" TIMESTAMP_DIR=False
#python ../ppo/ppo.py with EX_NAME="ppo_bc_test_unident_s" layout_name="unident_s" REW_SHAPING_HORIZON=6e6 PPO_RUN_TOT_TIMESTEPS=1e7 LR=1e-3 GPU_ID=1 OTHER_AGENT_TYPE="bc_test" SEEDS="[2888, 7424, 7360, 4467,  184]" VF_COEF=0.5 MINIBATCHES=12 LR_ANNEALING=3 SELF_PLAY_HORIZON="[1e6, 7e6]" TIMESTAMP_DIR=False
#
#python ../ppo/ppo.py with EX_NAME="ppo_bc_train_random1" layout_name="random1" REW_SHAPING_HORIZON=5e6 PPO_RUN_TOT_TIMESTEPS=1.6e7 LR=1e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456, 1887, 5578, 5987,  516]" VF_COEF=0.5 MINIBATCHES=15 LR_ANNEALING=1.5 SELF_PLAY_HORIZON="[2e6, 6e6]" TIMESTAMP_DIR=False
#python ../ppo/ppo.py with EX_NAME="ppo_bc_test_random1" layout_name="random1" REW_SHAPING_HORIZON=5e6 PPO_RUN_TOT_TIMESTEPS=1.6e7 LR=1e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_test" SEEDS="[2888, 7424, 7360, 4467,  184]" VF_COEF=0.5 MINIBATCHES=15 LR_ANNEALING=1.5 SELF_PLAY_HORIZON="[2e6, 6e6]" TIMESTAMP_DIR=False

#python ../ppo/ppo.py with EX_NAME="ppo_bc_train_random0_test1" layout_name="random0" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=9e6 LR=1.5e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456, 1887, 5578, 5987,  516]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=2 SELF_PLAY_HORIZON=None TIMESTAMP_DIR=True
#python ../ppo/ppo.py with EX_NAME="ppo_bc_test_random0_test1" layout_name="random0" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=9e6 LR=1.5e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_test" SEEDS="[2888, 7424, 7360, 4467,  184]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=2 SELF_PLAY_HORIZON=None TIMESTAMP_DIR=True

#python ../ppo/ppo.py with EX_NAME="ppo_bc_train_random3" layout_name="random3" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=1.2e7 LR=1.5e-3 GPU_ID=3 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456, 1887, 5578, 5987,  516]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=3 SELF_PLAY_HORIZON="[1e6, 4e6]" TIMESTAMP_DIR=False
#python ../ppo/ppo.py with EX_NAME="ppo_bc_test_random3" layout_name="random3" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=1.2e7 LR=1.5e-3 GPU_ID=3 OTHER_AGENT_TYPE="bc_test" SEEDS="[2888, 7424, 7360, 4467,  184]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=3 SELF_PLAY_HORIZON="[1e6, 4e6]" TIMESTAMP_DIR=False

#nice -n 2 python ../ppo/ppo.py with EX_NAME="err_pen_ppo_bc_train_random0_test2" layout_name="random0" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=9e6 LR=1.5e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456, 1887, 5578, 5987,  516]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=2 SELF_PLAY_HORIZON=None TIMESTAMP_DIR=True
#nice -n 2 python ../ppo/ppo.py with EX_NAME="err_pen_ppo_bc_test_random0_test2" layout_name="random0" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=9e6 LR=1.5e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_test" SEEDS="[2888, 7424, 7360, 4467,  184]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=2 SELF_PLAY_HORIZON=None TIMESTAMP_DIR=True
#

#nice -n 2 python ../ppo/ppo_strat_specific.py with EX_NAME="single_strat_wft_ppo_bc_train_random0_test4" layout_name="random0" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=9e6 LR=1.5e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456, 1887, 5578, 5987,  516]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=2 SELF_PLAY_HORIZON=None TIMESTAMP_DIR=True
#nice -n 2 python ../ppo/ppo_strat_specific.py with EX_NAME="single_strat_wft_ppo_bc_test_random0_test4" layout_name="random0" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=9e6 LR=1.5e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_test" SEEDS="[2888, 7424, 7360, 4467,  184]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=2 SELF_PLAY_HORIZON=None TIMESTAMP_DIR=True


#nice -n 2 python ../ppo/ppo_strat_specific.py with EX_NAME="dual_strat_wft_ppo_bc_train_random0_test5" layout_name="random0" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=9e6 LR=1.5e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456, 1887, 5578, 5987,  516]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=2 SELF_PLAY_HORIZON=None TIMESTAMP_DIR=True
#nice -n 2 python ../ppo/ppo_strat_specific.py with EX_NAME="dual_strat_wft_ppo_bc_test_random0_test5" layout_name="random0" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=9e6 LR=1.5e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_test" SEEDS="[2888, 7424, 7360, 4467,  184]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=2 SELF_PLAY_HORIZON=None TIMESTAMP_DIR=True
#


#nice -n 2 python ../ppo/ppo_strat_specific.py with EX_NAME="aa_strat3_ppo_bc_train_random3" layout_name="unident_s" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=1.2e7 LR=1.5e-3 GPU_ID=3 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456, 1887, 5578, 5987,  516]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=3 SELF_PLAY_HORIZON="[1e6, 4e6]" TIMESTAMP_DIR=True
#
#nice -n 2 python ../ppo/ppo_strat_specific.py with EX_NAME="aa_strat3_ppo_bc_test_random3" layout_name="unident_s" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=1.2e7 LR=1.5e-3 GPU_ID=3 OTHER_AGENT_TYPE="bc_test" SEEDS="[2888, 7424, 7360, 4467,  184]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=3 SELF_PLAY_HORIZON="[1e6, 4e6]" TIMESTAMP_DIR=True
#

#nice -n 2 python ../ppo/ppo_strat_specific.py with EX_NAME="h_proxy_DPstrat_6rew_handtuned_Handoff_weights_ppo_bc_train_random0_test6" layout_name="random0" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=9e6 LR=1.5e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456, 1887, 5578, 5987,  516]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=2 SELF_PLAY_HORIZON=None TIMESTAMP_DIR=True

#nice -n 2 python ../ppo/ppo_strat_specific.py with EX_NAME="STRATEXP_TEST1_random0_s1_weights_ppo_bc_train" layout_name="random0" STRATEGY_INDEX=1 REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=9e6 LR=1.5e-3 GPU_ID=2 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=2 SELF_PLAY_HORIZON=None TIMESTAMP_DIR=True
#nice -n 2 python ../ppo/ppo_strat_specific.py with EX_NAME="STRATEXP_TEST1_simple_s3_weights_ppo_bc_train" layout_name="simple" STRATEGY_INDEX=3 REW_SHAPING_HORIZON=1e6 PPO_RUN_TOT_TIMESTEPS=8e6 LR=1e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456]" VF_COEF=0.5 MINIBATCHES=10 LR_ANNEALING=3 SELF_PLAY_HORIZON="[5e5, 3e6]" TIMESTAMP_DIR=True

nice -n 1 python ../ppo/ppo_strat_specific.py with EX_NAME="STRATEXP_TEST1_unident_s0_weights_ppo_bc_train" layout_name="unident_s" STRATEGY_INDEX=0 REW_SHAPING_HORIZON=6e6 PPO_RUN_TOT_TIMESTEPS=1e7 LR=1e-3 GPU_ID=3 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456]" VF_COEF=0.5 MINIBATCHES=12 LR_ANNEALING=3 SELF_PLAY_HORIZON="[1e6, 7e6]" TIMESTAMP_DIR=True

#nice -n 1 python ../ppo/ppo_strat_specific.py with EX_NAME="STRATEXP_TEST1_random1_s1_weights_ppo_bc_train" layout_name="random1" STRATEGY_INDEX=1 REW_SHAPING_HORIZON=5e6 PPO_RUN_TOT_TIMESTEPS=1.6e7 LR=1e-3 GPU_ID=1 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456]" VF_COEF=0.5 MINIBATCHES=15 LR_ANNEALING=1.5 SELF_PLAY_HORIZON="[2e6, 6e6]" TIMESTAMP_DIR=True



