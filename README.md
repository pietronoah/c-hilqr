# Constrained HiLQR

This repository contains the code for the publication *"Constrained Trajectory Optimization for Hybrid Dynamical Systems"* by Pietro Noah Crestaz, Gökhan Alcan, and Ville Kyrki. The code implements a constrained version of the Hybrid iterative Linear Quadratic Regulator (HiLQR) algorithm, designed to optimize trajectories for hybrid dynamical systems while respecting constraints such as obstacle avoidance and input limits. 

## Overview

Constrained HiLQR extends the standard HiLQR approach by incorporating two constraint-handling mechanisms: **Discrete Barrier States (DBaS)** and **Augmented Lagrangian (AL)** methods. 

Key features include:
- Constraint handling for hybrid system trajectory optimization.
- Two constraint-handling approaches:
  - **DBaS**: An interior point method that prioritizes safety constraints directly.
  - **AL**: A penalty-based approach that adapts constraints to allow tighter navigation.
- Open-source implementation of core methods used for simulations and performance comparisons.

## Environment installation

1. Clone the repository.
    ```bash
    git clone https://github.com/pietronoah/c-hilqr
    cd c-hilqr/
    ```
   
2. Create and activate the Conda environment.
    ```bash
    conda env create -f environment.yml
    conda activate chilqr
    ```
   
3. Run the environment test file to verify setup.
    ```bash
    python environment_test.py
    ```

## Usage

You can modify the main parameters of the problem directly within the test files. Key variables include:
- **`x_start`**: Initial state of the system.
- **`state_goal`**: Desired goal state.
- **`Q`, `R`, `QT`**: Weight matrices for cost function terms. The last term in `Q` and `QT` represents the barrier state cost weight.
- **`h`**: Horizon length.
- **`dt`**: Step size.
- **`u_lim`**: Control input limits, structured as `[[u1_low, u1_up], [u2_low, u2_up], ...]`.
- **`obstacle_info`**: Array of obstacles, supporting:
  - **Circle**: `[xc, yc, r, 'circle']`
  - **Ellipse**: `[xc, yc, a, b, 'ellipse']`
  - **Square**: `[xc, yc, l, 'square']`

To run the solver:
```bash
python bouncing_ball_AL.py
```

# Citation

If you find this repository helpful in your research, please cite the corresponding paper:

```bibtex
@misc{crestaz2024constrainedtrajectoryoptimizationhybrid,
      title={Constrained Trajectory Optimization for Hybrid Dynamical Systems}, 
      author={Pietro Noah Crestaz and Gokhan Alcan and Ville Kyrki},
      year={2024},
      eprint={2410.22894},
      archivePrefix={arXiv},
      primaryClass={eess.SY},
      url={https://arxiv.org/abs/2410.22894}, 
}
