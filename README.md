# SwarmAdaptivePattern_GECCO25
This project implements the evolutionary algorithm **CMA-ES to evolve the optimal robot controller** (D), enabling it to **reproduce a phenotypic pattern** at the swarm scale, matching the **visual display target** (A). The swarm of **moving robots** is modeled in an original **sliding-puzzle setup** (C) — a grid where each cell can either be occupied by an autonomous robot or remain empty. This **distributed system** is characterized by two parameters: **density**, which refers to the number of robots relative to the total available positions in the grid; and **fluidity**, which represents the probability of an agent moving to a nearby empty position at each time step. **Communication between robots is local**, occurring through **signal exchanges** in a **neighbor-to-neighbor von Neumann topology** (D).

> **To cite this work**: Alessia Loi and Nicolas Bredeche. 2025. Evolving Neural Controllers for Adaptive Visual Pattern Formation by a Swarm of Robots. In Proceedings of GECCO 2025 @ Málaga (hybrid) The Genetic and Evolutionary Computation Conference (GECCO ’25). ACM, New York, NY, USA, 9 pages

> **Acknowledgement**: This work is supported by the Agence Nationale pour la Recherche under Grant ANR-24-CE33-7791

![Research summary](src/teaser_V2.png)

## **To run an experiment: Learning, Swarm (generalization), and Plots**

-   Set the learning and generalization parameters by modifying <u>learning_params.json</u> and <u>swarm_params.json</u>. The parameters are already settled to reproduce the paper results.
-   In the Bash script <u>launch.sh</u>, uncomment the lines that you wish to run. If you run <u>learning_main.py</u> only, you can later execute <u>swarm_main.py</u> or the analysis scripts by commenting out <u>learning_main.py</u> and filling the <u>learning_analysis_dir</u> and <u>swarm_analysis_dir</u> fields.
-   From the *src* directory, execute the Bash script <u>launch.sh</u>:
    `~/SwarmAdaptivePattern_GECCO25/src$ ./launch.sh` 
-   The results will be stored in a newly created <u>simulationAnalysis</u> folder.


## Details
- In the function <u>plot_flag</u> in <u>environments.py</u>: the plot of the signal pattern does not work for <u>grid_nb_rows</u> ≤ 10 and <u>grid_nb_cols</u> ≤ 10 because signal values can be negative. **Easy fix**: comment out the special debug display for small grids.
- To reproduce the paper results, set:
  - "env_name": "sliding_puzzle_incremental"
  - "sliding_puzzle_incremental_switch_eval": 0
  - "sliding_puzzle_incremental_nb_deletions_units": null
  - "sliding_puzzle_incremental_nb_deletions_percent": [0.0, X], where X is the complement of density (i.e., 1.0 - density).
- In <u>learning_params.json</u>, the parameter <u>sliding_puzzle_incremental_nb_deletions_percent</u> corresponds to the complement of density (1.0 - density).
  Example 1: If density = 1.0 (meaning all grid positions are occupied by robots), then sliding_puzzle_incremental_nb_deletions_percent = 0.0.
  Example 2: If density = 0.8 (meaning 20% of the grid positions are empty), then sliding_puzzle_incremental_nb_deletions_percent = 0.2.