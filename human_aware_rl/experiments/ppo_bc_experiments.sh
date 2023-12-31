#nice -n 2 python ../ppo/ppo.py with EX_NAME="ppo_bc_train_simple_REPLICATE1" layout_name="simple" REW_SHAPING_HORIZON=1e6 PPO_RUN_TOT_TIMESTEPS=8e6 LR=1e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456, 1887, 5578, 5987,  516]" VF_COEF=0.5 MINIBATCHES=10 LR_ANNEALING=3 SELF_PLAY_HORIZON="[5e5, 3e6]" TIMESTAMP_DIR=True
#nice -n 2 python ../ppo/ppo.py with EX_NAME="ppo_bc_test_simple" layout_name="simple" REW_SHAPING_HORIZON=1e6 PPO_RUN_TOT_TIMESTEPS=8e6 LR=1e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_test" SEEDS="[2888, 7424, 7360, 4467,  184]" VF_COEF=0.5 MINIBATCHES=10 LR_ANNEALING=3 SELF_PLAY_HORIZON="[5e5, 3e6]" TIMESTAMP_DIR=True
#
#nice -n 2 python ../ppo/ppo.py with EX_NAME="ppo_bc_train_unident_s_REPLICATE1" layout_name="unident_s" REW_SHAPING_HORIZON=6e6 PPO_RUN_TOT_TIMESTEPS=1e7 LR=1e-3 GPU_ID=1 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456, 1887, 5578, 5987,  516]" VF_COEF=0.5 MINIBATCHES=12 LR_ANNEALING=3 SELF_PLAY_HORIZON="[1e6, 7e6]" TIMESTAMP_DIR=True
#nice -n 2 python ../ppo/ppo.py with EX_NAME="ppo_bc_test_unident_s_REPLICATE1" layout_name="unident_s" REW_SHAPING_HORIZON=6e6 PPO_RUN_TOT_TIMESTEPS=1e7 LR=1e-3 GPU_ID=1 OTHER_AGENT_TYPE="bc_test" SEEDS="[2888, 7424, 7360, 4467,  184]" VF_COEF=0.5 MINIBATCHES=12 LR_ANNEALING=3 SELF_PLAY_HORIZON="[1e6, 7e6]" TIMESTAMP_DIR=True
#
#nice -n 2 python ../ppo/ppo.py with EX_NAME="ppo_bc_train_random1_REPLICATE1" layout_name="random1" REW_SHAPING_HORIZON=5e6 PPO_RUN_TOT_TIMESTEPS=1.6e7 LR=1e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456, 1887, 5578, 5987,  516]" VF_COEF=0.5 MINIBATCHES=15 LR_ANNEALING=1.5 SELF_PLAY_HORIZON="[2e6, 6e6]" TIMESTAMP_DIR=True
#nice -n 2 python ../ppo/ppo.py with EX_NAME="ppo_bc_test_random1_REPLICATE1" layout_name="random1" REW_SHAPING_HORIZON=5e6 PPO_RUN_TOT_TIMESTEPS=1.6e7 LR=1e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_test" SEEDS="[2888, 7424, 7360, 4467,  184]" VF_COEF=0.5 MINIBATCHES=15 LR_ANNEALING=1.5 SELF_PLAY_HORIZON="[2e6, 6e6]" TIMESTAMP_DIR=True
#
#nice -n 2 python ../ppo/ppo.py with EX_NAME="ppo_bc_train_random0_test2_REPLICATE1" layout_name="random0" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=9e6 LR=1.5e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456, 1887, 5578, 5987,  516]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=2 SELF_PLAY_HORIZON=None TIMESTAMP_DIR=True
#nice -n 2 python ../ppo/ppo.py with EX_NAME="ppo_bc_test_random0_test2_REPLICATE1" layout_name="random0" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=9e6 LR=1.5e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_test" SEEDS="[2888, 7424, 7360, 4467,  184]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=2 SELF_PLAY_HORIZON=None TIMESTAMP_DIR=True
#
#nice -n 2 python ../ppo/ppo.py with EX_NAME="ppo_bc_train_random3_REPLICATE1" layout_name="random3" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=1.2e7 LR=1.5e-3 GPU_ID=3 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456, 1887, 5578, 5987,  516]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=3 SELF_PLAY_HORIZON="[1e6, 4e6]" TIMESTAMP_DIR=True
#nice -n 2 python ../ppo/ppo.py with EX_NAME="ppo_bc_test_random3_REPLICATE1" layout_name="random3" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=1.2e7 LR=1.5e-3 GPU_ID=3 OTHER_AGENT_TYPE="bc_test" SEEDS="[2888, 7424, 7360, 4467,  184]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=3 SELF_PLAY_HORIZON="[1e6, 4e6]" TIMESTAMP_DIR=True

#nice -n 2 python ../ppo/ppo.py with EX_NAME="err_pen_ppo_bc_train_random0_test2" layout_name="random0" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=9e6 LR=1.5e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456, 1887, 5578, 5987,  516]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=2 SELF_PLAY_HORIZON=None TIMESTAMP_DIR=True
#nice -n 2 python ../ppo/ppo.py with EX_NAME="err_pen_ppo_bc_test_random0_test2" layout_name="random0" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=9e6 LR=1.5e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_test" SEEDS="[2888, 7424, 7360, 4467,  184]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=2 SELF_PLAY_HORIZON=None TIMESTAMP_DIR=True


#nice -n 2 python ../ppo/ppo_strat_specific.py with EX_NAME="dual_strat_nft_ppo_bc_train_random0_test3" layout_name="random0" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=9e6 LR=1.5e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456, 1887, 5578, 5987,  516]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=2 SELF_PLAY_HORIZON=None TIMESTAMP_DIR=True
#nice -n 2 python ../ppo/ppo_strat_specific.py with EX_NAME="dual_strat_nft_ppo_bc_test_random0_test3" layout_name="random0" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=9e6 LR=1.5e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_test" SEEDS="[2888, 7424, 7360, 4467,  184]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=2 SELF_PLAY_HORIZON=None TIMESTAMP_DIR=True


nice -n 2 python ../ppo/ppo.py with EX_NAME="server_side_specific_ppo_bc_train_random0_test2" layout_name="random0" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=9e6 LR=1.5e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456, 1887, 5578, 5987,  516]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=2 SELF_PLAY_HORIZON=None TIMESTAMP_DIR=True
nice -n 2 python ../ppo/ppo.py with EX_NAME="server_side_specific_ppo_bc_test_random0_test2" layout_name="random0" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=9e6 LR=1.5e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_test" SEEDS="[2888, 7424, 7360, 4467,  184]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=2 SELF_PLAY_HORIZON=None TIMESTAMP_DIR=True


nice -n 2 python ../ppo/ppo_strat_specific.py with EX_NAME="server_side_specific_dual_strat_nft_ppo_bc_train_random0_test3" layout_name="random0" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=9e6 LR=1.5e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456, 1887, 5578, 5987,  516]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=2 SELF_PLAY_HORIZON=None TIMESTAMP_DIR=True
nice -n 2 python ../ppo/ppo_strat_specific.py with EX_NAME="server_side_specific_dual_strat_nft_ppo_bc_test_random0_test3" layout_name="random0" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=9e6 LR=1.5e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_test" SEEDS="[2888, 7424, 7360, 4467,  184]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=2 SELF_PLAY_HORIZON=None TIMESTAMP_DIR=True





