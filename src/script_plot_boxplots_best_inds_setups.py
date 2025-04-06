import os
import json
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend instead of QtAgg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np



def script_plot_boxplots_best_inds_setups(experiences_dir_to_plot_list):

    all_dirs_rows = []
    for exp_id, dir in enumerate(experiences_dir_to_plot_list):

        # Get parameters from config files
        with open(f"{dir}/learning/learning_params.json", "r") as f:
            learning_params = json.load(f)

        if learning_params['evolutionary_settings']['env_name'] != "sliding_puzzle_incremental":
            print(f"plot_sliding_puzzle_incremental_learning_density_fluidity stopped because {dir} env_name is {learning_params['evolutionary_settings']['env_name']} and not 'sliding_puzzle_incremental'")
            exit()
        
        if learning_params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_switch_eval'] > 13000: # be careful with this line, 13000 is an arbitrary value
            phase = 1
        else:
            phase = 2 # normally, we look for the best individual at the end of learning

        p_move = round( learning_params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_proba_move'], 2) # fluidity ticks
        density = round( (learning_params['grid']['grid_size'] - learning_params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_nb_deletions_ticks'][phase-1] ) / learning_params['grid']['grid_size'], 2) # density ticks

        if density == 1.0:
            p_move = 0.0 # if density is max, agents can't move

        nb_runs = learning_params['evolutionary_settings']['nb_runs']

        dataset = pd.read_csv(f"{dir}/learning/data_all_runs/data_evo_all_runs_best_ind_per_run_per_phase.csv")
        dataset = dataset[dataset.Learning_phase==phase]
        
        for i, row in enumerate(dataset.itertuples(index=False)):  # index=False to skip the DataFrame index
            all_dirs_rows.append({
                'Exp_id': exp_id,
                'Run': row.Run,
                'Fitness': row.Fitness,
                'Label': f"$\\rho$={density}\n$\\Phi$={p_move}"
            })

        i += 1
        assert i == nb_runs, f"(i={i}) != (nb_runs={nb_runs})"

    data = pd.DataFrame(all_dirs_rows)

    plt.figure(figsize=(12, 7))
    sns.set_theme(style='darkgrid')
    _, ax = plt.subplots()
    sns.boxplot(x='Label',
                y='Fitness',
                data=data,
                color='skyblue',
                medianprops={'color': 'red', 'linewidth': 0.5},
                width=0.7,
                ax=ax)

    exp_setup_and_dims = dir.split('_')
    plt.xlabel("Sliding-puzzle setups", fontsize=12)
    plt.ylabel("Flags distance", fontsize=12)
    plt.ylim(-0.1, 1.1) # 0 and 1 are respectively min and max values of flag distance (fitness)
    plt.title(f"Best individuals across sliding-puzzle experiments ($\\rho$, $\\Phi$)\n{exp_setup_and_dims[-2]} {exp_setup_and_dims[-1]}, {nb_runs} runs", fontsize=14)

    data.to_csv(f"simulationAnalysis/plot_sliding_puzzle_incremental_{exp_setup_and_dims[-2]}_{exp_setup_and_dims[-1]}_learning_boxplots_best_inds_setups.csv") # write data
    plt.savefig(f"simulationAnalysis/plot_sliding_puzzle_incremental_{exp_setup_and_dims[-2]}_{exp_setup_and_dims[-1]}_learning_boxplots_best_inds_setups.png")

    plt.clf()
    plt.close()






experiences_dir_to_plot_list = [
    "/home/kalessia/SwarmAdaptivePattern_GECCO25/src/simulationAnalysis/sliding_puzzle_incremental_2025-03-26_00-54-56_two-bands_16x16"
]

script_plot_boxplots_best_inds_setups(experiences_dir_to_plot_list)