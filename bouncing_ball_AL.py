import numpy as np
import torch
import matplotlib.pyplot as plt
import traceback

from os import X_OK
from torch.autograd.functional import jacobian, hessian
from torch.autograd import grad
from systems.systems_AL import BouncingBall2D_AL
from solvers.AL_HiLQR import *
from utils import *

import warnings
warnings.filterwarnings("ignore")


def main():
    try:
        # Initialize system dynamics
        system = BouncingBall2D_AL()

        # Define goal state
        x_start = torch.tensor([0., 4., 0., 0.])
        x_goal = torch.tensor([10., 1., 0., 0.])
        
        # Define weight matrices
        Q = torch.diag(torch.tensor([0., 0., 0., 0.]))
        R = torch.diag(torch.tensor([0.005, 0.005]))
        QT = torch.diag(torch.tensor([4000., 4000., 0., 0.]))

        # Time step and horizon length
        h = 200
        dt = 0.02
        
        # Control input saturation
        u_lim = [[-10,10],
                 [-10,10]]
        
        # Obstacles' info
        obstacle_info = []

        obstacle_info = [[5.5,0,0.75,0.05,'ellipse'],
                        [3,3,0.5,'circle'],
                        [6,2.5,0.7,'circle'],
                        [9,0,0.2,0.05,'ellipse'],
                        [3,1.5,0.25,'circle'],
                        [6.5,1.5,1,'square']]
        
        # Fix system properties
        system.set_cost(Q, R)
        system.set_final_cost(QT)
        system.set_goal(x_goal)
        system.set_dt(dt)
        system.set_control_limits(u_lim)
        system.set_obstacles(obstacle_info)
        system.set_horizon(h)
        
        # Initial guess for the control input
        u_init = torch.ones((h, system.control_size)) * 0.2
        
        # Initialize the solver
        solver = AL_HiLQR(system, x_start, u_init, horizon=h)
        
        # Solve the control problem
        solver.fit()
        
        # Plot results
        plotter2DBall(system, solver, h)
        #animation2DBall(system, solver, h)

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()

