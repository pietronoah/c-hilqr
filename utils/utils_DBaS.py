import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import matplotlib.animation as animation
from IPython import display 
import textwrap
from pyfiglet import Figlet

import os

import torch

plt.style.use('seaborn-v0_8-bright')

def plotter2DBall(system, solver, h):
    
    # Plot theta and action trajectory
    figure, axis = plt.subplots(3, 2)
        
    axis[0, 0].add_patch(plt.Rectangle((-2, -2), system.goal[0]+4, 2, color='gray', alpha=0.5))
    
    # Patch color preferences
    color_patch = '#c23b22'
    alpha_patch = 0.6
    
    for i in range(len(system.obstacle_info)):
        match system.obstacle_info[i][-1]:
            case 'circle':
                obstacle_patch = plt.Circle((system.obstacle_info[i][0], system.obstacle_info[i][1]), system.obstacle_info[i][2], color=color_patch, alpha=alpha_patch)
            case 'ellipse':
                obstacle_patch = Ellipse(xy=(system.obstacle_info[i][0], system.obstacle_info[i][1]), width = 2*system.obstacle_info[i][2], height = 2*system.obstacle_info[i][3], color=color_patch, alpha=alpha_patch)
            case 'square':
                obstacle_patch = plt.Rectangle((system.obstacle_info[i][0]-system.obstacle_info[i][2]/2, system.obstacle_info[i][1]-system.obstacle_info[i][2]/2), system.obstacle_info[i][2], system.obstacle_info[i][2], color=color_patch, alpha=alpha_patch)
        axis[0, 0].add_patch(obstacle_patch)


    axis[0, 0].plot(solver.rollout_x[:, 0], solver.rollout_x[:, 1], color='black', linestyle='dashed', linewidth=0.5, label='Start rollout')    
    axis[0, 0].plot(solver.x_trajectories[:, 0], solver.x_trajectories[:, 1],label="DBaS solution")#, color='tomato')
    axis[0, 0].plot(solver.x_trajectories[-1, 0], solver.x_trajectories[-1, 1], 'o')
    axis[0, 0].scatter(system.goal[0],system.goal[1], marker='*', color='gold', edgecolor='black', linewidth=0.5, zorder=6)
    axis[0, 0].grid(True)
    axis[0, 0].set_xlabel('y [m]')
    axis[0, 0].set_ylabel("z [m]")
    axis[0, 0].axis('equal')
    axis[0, 0].set_xlim([0, 10])
    axis[0, 0].legend()

    final_cost = solver.get_final_cost()
    figure.suptitle('Number of iterations: ' + str(solver.niter) + ', J: ' + str(format_number(final_cost.numpy().item())), fontsize=12)

    axis[0, 1].plot(np.linspace(0, h * system.dt, num=h + 1), solver.x_trajectories[:, 2],label='Vy')#, color='tomato')
    axis[0, 1].plot(np.linspace(0, h * system.dt, num=h + 1), solver.x_trajectories[:, 3],label='Vz')#, color="Blue")
    axis[0, 1].grid(True)
    axis[0, 1].set_ylabel("Velocity [m/s]")
    axis[0, 1].set_xlabel('Time [s]')
    axis[0, 1].legend()

    axis[1, 0].plot(np.linspace(0, h * system.dt, num=h), solver.u_trajectories[:, 0].detach(), label='Fy')#, color='tomato')
    axis[1, 0].plot(np.linspace(0, h * system.dt, num=h), solver.u_trajectories[:, 1].detach(), label='Fz')#, color="Blue")
    axis[1, 0].grid(True)
    axis[1, 0].set_xlabel('Time [s]')
    axis[1, 0].set_ylabel("Input [N]")
    axis[1, 0].legend()

    axis[1, 1].plot(np.linspace(0, h * system.dt, num=h+1), solver.x_trajectories[:, 0],label='py') 
    axis[1, 1].plot(np.linspace(0, h * system.dt, num=h+1), solver.x_trajectories[:, 1],label='pz') 
    axis[1, 1].grid(True)
    axis[1, 1].set_xlabel('Time [s]')
    axis[1, 1].set_ylabel("py")
    axis[1, 1].legend()

    delta_u = torch.zeros((solver.niter,))
    for i in range(solver.niter):
        temp = solver.u_trajectories_history[:,:,i] - solver.u_trajectories[:,:]
        delta_u[i] = torch.norm(temp)

    axis[2, 0].plot(np.linspace(1, solver.niter, num=solver.niter), delta_u, marker='*',label='Δu') 
    axis[2, 0].grid(True)
    axis[2, 0].set_xlabel('Iteration')
    axis[2, 0].set_ylabel("Δu")
    axis[2, 0].legend()

    axis[2, 1].plot(np.linspace(1, solver.niter, num=solver.niter), solver.J_history[:solver.niter], marker='*')#, "Blue")
    axis[2, 1].grid(True)
    axis[2, 1].set_xlabel('Iteration')
    axis[2, 1].set_ylabel("J")


    plt.show()
    
    
    
    
def animation2DBall(system, solver, h):
    
    # Plot theta and action trajectory
    figure, axis = plt.subplots()
        
    axis.add_patch(plt.Rectangle((-1, -2), system.goal[0]+2, 2, color='gray', alpha=0.5))
    
    # Patch color preferences
    color_patch = '#c23b22'
    alpha_patch = 0.6
    
    for i in range(len(system.obstacle_info)):
        match system.obstacle_info[i][-1]:
            case 'circle':
                obstacle_patch = plt.Circle((system.obstacle_info[i][0], system.obstacle_info[i][1]), system.obstacle_info[i][2], color=color_patch, alpha=alpha_patch)
            case 'ellipse':
                obstacle_patch = Ellipse(xy=(system.obstacle_info[i][0], system.obstacle_info[i][1]), width = 2*system.obstacle_info[i][2], height = 2*system.obstacle_info[i][3], color=color_patch, alpha=alpha_patch)
            case 'square':
                obstacle_patch = plt.Rectangle((system.obstacle_info[i][0]-system.obstacle_info[i][2]/2, system.obstacle_info[i][1]-system.obstacle_info[i][2]/2), system.obstacle_info[i][2], system.obstacle_info[i][2], color=color_patch, alpha=alpha_patch)
        axis.add_patch(obstacle_patch)
    
    anim = axis.plot(solver.x_trajectories_history[:, 0, 0], solver.x_trajectories_history[:, 1, 0], label='new traj')[0]#, color='tomato')
    anim_old = axis.plot(solver.x_trajectories_history[:, 0, 0], solver.x_trajectories_history[:, 1, 0], label='old traj')[0]#, color='tomato')
    #end_anim = axis.plot(solver.x_trajectories_history[-1, 0, 0], solver.x_trajectories_history[-1, 1, 0], 'o')[0]
    axis.scatter(system.goal[0],system.goal[1], marker='*', color='gold', edgecolor='black', linewidth=0.5, zorder=6)
    axis.grid(True)
    axis.set_xlabel('x')
    axis.set_ylabel("y")
    axis.axis('equal')
    axis.legend()
    
    def update(frame):
        step = frame%(h+1)
        iter = int((frame-step)/(h+1))
        # for each frame, update the data stored on each artist.
        trajx = solver.x_trajectories_history[:step, 0, iter+1]
        trajy = solver.x_trajectories_history[:step, 1, iter+1]
        
        trajx_old = solver.x_trajectories_history[:, 0, iter]
        trajy_old = solver.x_trajectories_history[:, 1, iter]
            
        anim.set_xdata(trajx)
        anim.set_ydata(trajy)
        
        anim_old.set_xdata(trajx_old)
        anim_old.set_ydata(trajy_old)
            
        return(anim,anim_old)

    ani = animation.FuncAnimation(fig=figure, func=update, frames=(h+1)*solver.niter, interval=20, blit=True)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('ani.mp4', writer=writer)
    
def animation2DBall_low_lr_input(system, solver, h):
    
    # Plot theta and action trajectory
    figure, axis = plt.subplots()
    u1 = axis.plot(np.linspace(0, h * system.dt, num=h), solver.u_trajectories_low_lr_history[:, 0,0].detach(), label='Fy')[0]
    u2 = axis.plot(np.linspace(0, h * system.dt, num=h), solver.u_trajectories_low_lr_history[:, 1,0].detach(), label='Fz')[0]
    axis.grid(True)
    axis.set_xlabel('Time[s]')
    axis.set_ylabel("Force")
    #axis.axis('equal')
    axis.legend()
    
    def update(frame):
        """ step = frame%(h)
        iter = int((frame-step)/(h)) """
        u1_t = solver.u_trajectories_low_lr_history[:, 0, frame]
        u2_t = solver.u_trajectories_low_lr_history[:, 1, frame]
            
        u1.set_xdata(np.linspace(0, h * system.dt, num=h))
        u1.set_ydata(u1_t)
        u2.set_xdata(np.linspace(0, h * system.dt, num=h))
        u2.set_ydata(u2_t)
        
        return(u1, u2)

    ani = animation.FuncAnimation(fig=figure, func=update, frames=20, interval=1000, blit=True)
    
    plt.show()
    
def animation2DBall_low_lr_state(system, solver, h):
    
    figure, axis = plt.subplots()
        
    axis.add_patch(plt.Rectangle((0, -2), system.goal[0], 2, color='gray', alpha=0.5))
    
    # Patch color preferences
    color_patch = '#c23b22'
    alpha_patch = 0.6
    
    for i in range(len(system.obstacle_info)):
        match system.obstacle_info[i][-1]:
            case 'circle':
                obstacle_patch = plt.Circle((system.obstacle_info[i][0], system.obstacle_info[i][1]), system.obstacle_info[i][2], color=color_patch, alpha=alpha_patch)
            case 'ellipse':
                obstacle_patch = Ellipse(xy=(system.obstacle_info[i][0], system.obstacle_info[i][1]), width = 2*system.obstacle_info[i][2], height = 2*system.obstacle_info[i][3], color=color_patch, alpha=alpha_patch)
            case 'square':
                obstacle_patch = plt.Rectangle((system.obstacle_info[i][0]-system.obstacle_info[i][2]/2, system.obstacle_info[i][1]-system.obstacle_info[i][2]/2), system.obstacle_info[i][2], system.obstacle_info[i][2], color=color_patch, alpha=alpha_patch)
        axis.add_patch(obstacle_patch)
        
    anim = axis.plot(solver.x_trajectories_low_lr_history[:, 0, 0], solver.x_trajectories_low_lr_history[:, 1, 0], label='new traj')[0]
    axis.scatter(system.goal[0],system.goal[1], marker='*', color='gold', edgecolor='black', linewidth=0.5, zorder=6)
    axis.grid(True)
    axis.set_xlabel('x')
    axis.set_ylabel("y")
    axis.axis('equal')
    axis.legend()
    
    def update(frame):
        
        trajx = solver.x_trajectories_low_lr_history[:, 0, frame]
        trajy = solver.x_trajectories_low_lr_history[:, 1, frame]
            
        anim.set_xdata(trajx)
        anim.set_ydata(trajy)
            
        return(anim)

    ani = animation.FuncAnimation(fig=figure, func=update, frames=20, interval=1000)#, blit=True)
    
    plt.show()
    
    
def format_number(num):
    if round(num, 4) == 0:
        return '{:.4e}'.format(num)
    else:
        return '{:.4f}'.format(num)
    
    
def format_large_number(num):
    if abs(num) >= 1000:
        return '{:.2e}'.format(num)
    else:
        return '{:.4f}'.format(num)
    
    
    
# Function to format and print the header
def print_header():
    custom_fig = Figlet(font='standard') #doom

    # Render the text
    text = custom_fig.renderText('DBaS-HiLQR Optimizer')

    # Print the rendered text
    print()
    print(text)
    print()
    header = ("iter", "objective", "ΔJ", "||w||", "||e||", "Alpha", "LS", "CV")
    print(format_row(header))

# Function to format each row
def format_row(row):
    formatted_row = ""
    for item, width in zip(row, column_widths):
        formatted_row += str(item).ljust(width) + " | "  # Convert item to string before applying ljust
    return formatted_row

# Maximum width for each column
column_widths = [6, 10, 10, 10, 10, 10, 6, 6]

def print_current_iteration(iteration, objective, delta_J, w, error, alpha, ls, violation):
    current_row = (iteration, objective, delta_J, w, error, alpha, ls, str(int(violation*100))+"%")
    print(format_row(current_row))