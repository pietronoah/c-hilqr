# Created by Pietro Noah Crestaz
# Aalto University

# Library import
from os import X_OK
import numpy as np
from torch.autograd.functional import jacobian, hessian
from torch.autograd import grad
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
from utils.utils_DBaS import * 


# Define the main class iLQR
class DBaS_HiLQR:
    def __init__(self, system, initial_state, u_init, horizon = 300):
        self.system = system
        self.horizon = horizon
        self.initial_state = initial_state
        self.x_trajectories = torch.zeros((self.horizon + 1, self.system.state_size))
        self.x_trajectories[0,:] = self.initial_state
        self.x_trajectories_history = torch.zeros((self.horizon + 1, self.system.state_size, 100))
        self.u_trajectories = u_init
        self.u_trajectories_history = torch.zeros((self.horizon, self.system.control_size, 100))
        self.x_trajectories_low_lr_history = torch.zeros((self.horizon+1, self.system.state_size, 30))
        self.u_trajectories_low_lr_history = torch.zeros((self.horizon, self.system.control_size, 30))
        self.ks = torch.zeros((self.horizon, self.system.control_size))
        self.Ks = torch.zeros((self.horizon, self.system.control_size, self.system.state_size))
        self.J_new = 10000
        self.J = 1000
        self.J_history = torch.zeros((100,))
        self.deltaV = 0
        self.alphas = 0.4**np.arange(20) # Line search candidates
        self.dalpha1 = 0
        self.dalpha2 = 0
        self.niter = 0
        self.concluded = 0
        self.tolerance = 1e-2
        self.max_iter = 30
        self.mode = self.system.init_mode
        
        self.low_lr = 0

        self.state = torch.zeros((self.system.state_size,1))
        self.modes = np.zeros(self.horizon + 1, dtype=int)
        self.modes[0] = self.system.init_mode

        self.impact = np.zeros(self.horizon, dtype=int)
        self.impact_ = np.zeros(self.horizon, dtype=int)

        self.delta_u_1 = torch.zeros((self.horizon, self.system.control_size))
        self.delta_u_2 = torch.zeros((self.horizon, self.system.control_size))

        # Hybrid storage
        self.impact_states_ = np.empty((0,self.system.state_size))
        self.reset_states_ = np.empty((0,self.system.state_size))
        self.impact_mode_vec_ = np.empty((0,1), dtype=int)
        self.reset_mode_vec_ = np.empty((0,1), dtype=int)
        self.impact_idx_vec_ = np.empty((0,1), dtype=int)
        self.transition_inputs_ = np.empty((0,self.system.control_size))
        self.hybrid_transitions_ = 0

        # Expected cost reductions
        self.expected_cost_redu_ = 0
        self.expected_cost_redu_grad_ = 0
        self.expected_cost_redu_hess_ = 0

        self.unfeasible_start = 0

    def fit(self):
        self.system.CV = 0
        print()
        print_header()
        
        self.rollout()
        if self.system.CV == 1:
            print()
            print("Unfeasible initial trajectory")
            print()
            self.unfeasible_start = 1
        self.rollout_x = self.x_trajectories

        armijo_threshold = 0.1

        start = time.time()

        while self.concluded != 1 and self.niter < self.max_iter:
            self.backward_pass()
            
            self.system.CV = 0
            CV_n = 0
            CV_percentage = 0
            
            it_alpha = 0
            armijo_flag = 0
            while(it_alpha < len(self.alphas) and armijo_flag == 0):
                alpha = self.alphas[it_alpha]
                storage = self.forward_pass(alpha)
                cost_diff = self.J - self.J_new
                expected_cost_redu = alpha*self.expected_cost_redu_grad_ + pow(alpha,2)*self.expected_cost_redu_hess_
                
                if(cost_diff > 0 and self.system.CV==0):
                    armijo_flag = 1
                    self.J = self.J_new
                    self.x_trajectories = storage["x_new_trajectories"]
                    self.u_trajectories = storage["u_new_trajectories"]
                    self.modes = storage["modes"]
                    self.impact_states_ = storage["impact_states"]
                    self.reset_states_ = storage["reset_states"]
                    self.impact_mode_vec_ = storage["impact_mode_vec"]
                    self.reset_mode_vec_ = storage["reset_mode_vec"]
                    self.impact_idx_vec_ = storage["impact_idx_vec"]
                    self.transition_inputs_ = storage["transition_inputs"]
                    self.impact_ = self.impact
                    
                    end_error = np.linalg.norm(self.x_trajectories[-1,:2] - self.system.goal[:2])
                    w = np.linalg.norm(self.x_trajectories[:,self.system.state_size-1])
                    if it_alpha > 0:
                        CV_percentage = CV_n/it_alpha
                    else:
                        CV_percentage = 0
                    self.niter += 1
                    print_current_iteration(self.niter, format_large_number(self.J), format_large_number(-cost_diff.numpy()), format_large_number(w), format_large_number(end_error), format_number(self.alphas[it_alpha]), it_alpha+1, CV_percentage)
                else:
                    self.x_trajectories_low_lr_history[:,:,it_alpha] = storage["x_new_trajectories"]
                    self.u_trajectories_low_lr_history[:,:,it_alpha] = storage["u_new_trajectories"]
                    it_alpha += 1
                    if self.system.CV == 1:
                        CV_n += 1
                        self.system.CV = 0
    
            self.x_trajectories_history[:,:,self.niter-1] = self.x_trajectories
            self.u_trajectories_history[:,:,self.niter-1] = self.u_trajectories
            self.J_history[self.niter-1] = self.J

            if abs(cost_diff) < self.tolerance:
                print()
                print("Optimal trajectory found!")
                print("Optimal trajectory cost: ", float(self.J))
                end_error = np.linalg.norm(self.x_trajectories[-1,:2] - self.system.goal[:2])
                if end_error < 0.05:
                    self.convergency = 1
                else:
                    self.convergency = 0
                print("Norm of the error: ", end_error)
                end = time.time()
                print('Total time in the optimizer: ', end - start)
                print()
                
                return

            # If learning rate is low, then stop optimization
            if(alpha==self.alphas[-1]):
                print()
                print("Stopping optimization, low alpha")
                self.low_lr = 1
                end = time.time()
                print("Optimal trajectory cost: ", float(self.J))
                print('Total time in the optimizer: ', end - start)
                self.niter += 1
                self.convergency = 0
                return
            

        end = time.time()
        if self.niter == self.max_iter:
            print()
            print("Max number of iterations reached!")
            print("Optimal trajectory cost: ", float(self.J))
            print('Total time in the optimizer: ', end - start)
            self.convergency = 0
            
                

    def rollout(self):
        impact_states = np.empty((0,self.system.state_size))
        reset_states = np.empty((0,self.system.state_size))
        impact_mode_vec = np.empty((0,1), dtype=int)
        reset_mode_vec = np.empty((0,1), dtype=int)
        impact_idx_vec = np.empty((0,1), dtype=int)
        transition_inputs = np.empty((0,self.system.control_size))
        hybrid_transitions = 0

        self.J = 0.

        self.x_trajectories[0,:] = self.initial_state
        for i in range(self.horizon):
            storage = self.simulateHybridTimestep(self.x_trajectories[i,:], self.u_trajectories[i,:], i)
            impact_states = np.append(impact_states, storage["impact_states"])
            reset_states = np.append(reset_states, storage["reset_states"])
            impact_mode_vec = np.append(impact_mode_vec, storage["impact_mode_vec"])
            reset_mode_vec = np.append(reset_mode_vec, storage["reset_mode_vec"])
            impact_idx_vec = np.append(impact_idx_vec, storage["impact_idx_vec"])
            transition_inputs = np.append(transition_inputs, storage["transition_inputs"])
            hybrid_transitions = hybrid_transitions + storage["hybrid_transitions"]
            self.x_trajectories[i+1,:] = storage["next_state"]
            self.modes[i+1] = storage["next_mode"]

            self.J += self.system.L(self.x_trajectories[i,:], self.u_trajectories[i,:]) 

        self.impact_states_ = impact_states
        self.reset_states_ = reset_states
        self.impact_mode_vec_ = impact_mode_vec
        self.reset_mode_vec_ = reset_mode_vec
        self.impact_idx_vec_ = impact_idx_vec
        self.transition_inputs_ = transition_inputs
        self.hybrid_transitions_ = hybrid_transitions

        self.J += self.system.Lt(self.x_trajectories[-1,:])
        

    def backward_pass(self):
        self.delta_V = 0

        p,P = self.system.Lt_prime(self.x_trajectories[-1,:])
        
        self.dalpha1 = 0
        self.dalpha2 = 0

        expected_cost_redu = 0
        expected_cost_redu_grad = 0
        expected_cost_redu_hess = 0

        # Back propagate along the trajectory
        for i in range(self.horizon-1,-1,-1): 
            x = self.x_trajectories[i,:]
            u = self.u_trajectories[i,:]
            current_mode = self.modes[i]

            idx, = np.where(self.impact_idx_vec_ == i)

            if idx.size > 0:
                for j in range(idx.size-1,-1,-1):
                    impact_states = self.impact_states_[self.system.state_size*idx[j]:self.system.state_size*(idx[j]+1)]
                    impact_inputs = self.transition_inputs_[self.system.control_size*idx[j]:self.system.control_size*(idx[j]+1)]
                    impact_mode = self.impact_mode_vec_[idx[j]:(idx[j]+1)]
                    salt = self.system.salt[impact_mode[0]](torch.Tensor(impact_states),torch.Tensor(impact_inputs))
                    p = torch.matmul(torch.transpose(salt,0,1),p.reshape(1,-1).t()).squeeze().t()
                    P = torch.matmul(torch.matmul(torch.transpose(salt,0,1),P),salt)
                
                current_mode = impact_mode               

            f_x, f_u = self.system.f_prime(x, u)
            """ if i == 100:
                print(f_x) """
            start = time.time()
            l_x, l_u, l_xx, l_ux, l_uu = self.system.L_prime(x, u)
            end = time.time()
            
            Q_xx = l_xx + f_x.T@P@f_x
            Q_uu = l_uu + f_u.T@P@f_u
            Q_ux = l_ux + f_u.T@P@f_x
            Q_x = l_x + f_x.T@p
            Q_u = l_u + f_u.T@p

            self.ks[i] = -torch.linalg.inv(Q_uu)@Q_u
            self.Ks[i] = -torch.linalg.inv(Q_uu)@Q_ux

            P = Q_xx + self.Ks[i].T@Q_uu@self.Ks[i] + 2*self.Ks[i].T@Q_ux
            p = Q_x + self.Ks[i].T@Q_uu@self.ks[i] + self.Ks[i].T@Q_u + Q_ux.T@self.ks[i]

            self.delta_V += self.ks[i].T@Q_u + 0.5*self.ks[i].T@Q_uu@self.ks[i]

            self.dalpha1 += self.ks[i].T@Q_u
            self.dalpha2 += 0.5*self.ks[i].T@Q_uu@self.ks[i]

            # Get the current expected cost reduction from each source
            current_cost_reduction_grad = -Q_u.T@self.ks[i]
            current_cost_reduction_hess = 0.5 * self.ks[i].T@Q_uu@self.ks[i]
            current_cost_reduction = current_cost_reduction_grad + current_cost_reduction_hess

            expected_cost_redu_grad = expected_cost_redu_grad + current_cost_reduction_grad
            expected_cost_redu_hess = expected_cost_redu_hess + current_cost_reduction_hess
            expected_cost_redu = expected_cost_redu + current_cost_reduction

        self.expected_cost_redu_grad_ = expected_cost_redu_grad
        self.expected_cost_redu_hess_ = expected_cost_redu_hess
        self.expected_cost_redu_ = expected_cost_redu



    def forward_pass(self,alpha):
        self.mode = self.system.init_mode

        x_new_trajectories = torch.zeros((self.horizon + 1, self.system.state_size))
        u_new_trajectories = torch.zeros((self.horizon, self.system.control_size))

        current_state = self.initial_state
        current_mode = self.system.init_mode

        self.J_new = 0.
        x_new_trajectories[0,:] = self.x_trajectories[0,:]

        modes = np.zeros(self.horizon + 1, dtype=int)
        modes[0] = self.system.init_mode

        impact_states = np.empty((0,self.system.state_size))
        reset_states = np.empty((0,self.system.state_size))
        impact_mode_vec = np.empty((0,1), dtype=int)
        reset_mode_vec = np.empty((0,1), dtype=int)
        impact_idx_vec = np.empty((0,1), dtype=int)
        transition_inputs = np.empty((0,self.system.control_size))
        hybrid_transitions = 0


        for i in range(self.horizon):

            reference_hybrid_transitions = sum(i >= self.impact_idx_vec_)
            mode_count_difference = hybrid_transitions-reference_hybrid_transitions

            if modes[i] != self.modes[i]:
                
                if mode_count_difference <= 0: # Late impact
                    impact_idx = self.impact_idx_vec_[reference_hybrid_transitions-1]
                    ref_state = self.impact_states_[self.system.state_size*(reference_hybrid_transitions-1):self.system.state_size*reference_hybrid_transitions]
                    ref_input = self.transition_inputs_[reference_hybrid_transitions-1]
                    ref_k_feedforward = self.ks[impact_idx]
                    ref_K_feedback = self.Ks[impact_idx]
                    self.impact[i] = 0
                    
                else: # Early impact
                    
                    if(hybrid_transitions>len(self.impact_idx_vec_)):
                        ref_state = self.x_trajectories[self.horizon,:]
                        ref_input = self.u_trajectories[-1,:]
                        ref_k_feedforward = 0. * self.ks[-1]
                        ref_K_feedback = self.Ks[-1]
                        self.impact[i] = 1
                        
                    else:
                        impact_idx = self.impact_idx_vec_[hybrid_transitions-1]
                        ref_state = self.reset_states_[self.system.state_size*(hybrid_transitions-1):self.system.state_size*hybrid_transitions]
                        ref_input = self.transition_inputs_[hybrid_transitions-1]
                        ref_k_feedforward = self.ks[impact_idx]
                        ref_K_feedback = self.Ks[impact_idx]
                        self.impact[i] = 2

                delta_u = ref_K_feedback@(x_new_trajectories[i,:]-torch.Tensor(ref_state)) + alpha*ref_k_feedforward
                u_new_trajectories[i,:] = ref_input + delta_u
                
                # Control input saturation
                for it in range(len(u_new_trajectories[i,:])):
                    u_new_trajectories[i,it] = max(min(u_new_trajectories[i,it], self.system.u_lim[it][1]), self.system.u_lim[it][0])
                
                self.delta_u_1[i,:] = ref_K_feedback@(x_new_trajectories[i,:]-torch.Tensor(ref_state))
                self.delta_u_2[i,:] = ref_k_feedforward

            else:
                ref_k_feedforward = self.ks[i]
                ref_K_feedback = self.Ks[i]
                ref_state = self.x_trajectories[i,:]
                delta_u = ref_K_feedback@(x_new_trajectories[i,:]-ref_state) + alpha*ref_k_feedforward
                u_new_trajectories[i,:] = self.u_trajectories[i,:] + delta_u
                
                for it in range(len(u_new_trajectories[i,:])):
                    u_new_trajectories[i,it] = max(min(u_new_trajectories[i,it], self.system.u_lim[it][1]), self.system.u_lim[it][0])
                    
                self.impact[i] = -1
                self.delta_u_1[i,:] = ref_K_feedback@(x_new_trajectories[i,:]-torch.Tensor(ref_state))
                self.delta_u_2[i,:] = ref_k_feedforward

            storage = self.simulateHybridTimestep(x_new_trajectories[i,:], u_new_trajectories[i,:],i)

            x_new_trajectories[i+1,:] = storage["next_state"]
            modes[i+1] = storage["next_mode"]

            impact_states = np.append(impact_states, storage["impact_states"])
            reset_states = np.append(reset_states, storage["reset_states"])
            impact_mode_vec = np.append(impact_mode_vec, storage["impact_mode_vec"])
            reset_mode_vec = np.append(reset_mode_vec, storage["reset_mode_vec"])
            impact_idx_vec = np.append(impact_idx_vec, storage["impact_idx_vec"])
            transition_inputs = np.append(transition_inputs, storage["transition_inputs"])
            hybrid_transitions = hybrid_transitions + storage["hybrid_transitions"]

            self.J_new += self.system.L(x_new_trajectories[i,:], u_new_trajectories[i,:]) 

        self.J_new += self.system.Lt(x_new_trajectories[-1,:])

        storage = {"x_new_trajectories": x_new_trajectories, 
                   "u_new_trajectories": u_new_trajectories, 
                   "modes": modes,
                   "impact_states": impact_states, 
                   "reset_states": reset_states, 
                   "impact_mode_vec": impact_mode_vec, 
                   "reset_mode_vec": reset_mode_vec, 
                   "impact_idx_vec": impact_idx_vec, 
                   "transition_inputs": transition_inputs, 
                   "hybrid_transitions": hybrid_transitions}

        return storage
    
    def eventAttr():
        def decorator(func):
            func.terminal = True
            return func
        return decorator

    @eventAttr()
    def guardsCheck(self,t,x,u):
        a = self.system.g_ODE[self.mode](x,u) # Decision variable
        
        if not torch.matmul(self.system.g_prime[self.mode](torch.Tensor(x),torch.Tensor(u))[0][0],self.system.f(torch.Tensor(x),torch.Tensor(u))) > 0:
            a = -1
        return a

    def simulateHybridTimestep(self, current_state, current_input, iter):
        tspan = np.linspace(self.system.dt*iter, self.system.dt*(iter+1), num=2)

        impact_states = np.empty((0,self.system.state_size))
        reset_states = np.empty((0,self.system.state_size))
        impact_mode_vec = np.empty((0,1), dtype=int)
        impact_idx_vec = np.empty((0,1), dtype=int)
        reset_mode_vec = np.empty((0,1), dtype=int)
        transition_inputs = np.empty((0,self.system.control_size))
        hybrid_transitions = 0

        if self.system.g[self.mode](current_state,current_input)[0] > -1: # Check this condition only if close to constraint
            transverse_cond = 1 if (torch.matmul(self.system.g_prime[self.mode](current_state,current_input)[0][0],self.system.f(self.x_trajectories[iter,:],self.u_trajectories[iter,:]))) > 0 else 0
        else:
            transverse_cond = 0

        if transverse_cond == 1 and self.system.g[self.mode](current_state,current_input) >= 0:
            impact_states = np.append(impact_states,[current_state.numpy()], axis=0)
            current_state = self.system.Rm[self.mode](current_state,current_input)
            reset_states = np.append(reset_states,[current_state.numpy()], axis=0)
            impact_mode_vec = np.append(impact_mode_vec, self.mode)
            # This is just because we have two modes
            if self.mode == 0:
                self.mode = 1
            else:
                self.mode = 0
            impact_idx_vec = np.append(impact_idx_vec, iter)
            reset_mode_vec = np.append(reset_mode_vec, self.mode)
            transition_inputs = np.append(transition_inputs,[current_input.numpy()], axis=0)
            hybrid_transitions += 1

        # ODE time - here we could add a control to check if we are far from constraints, and so avoid the numerical integraiton
        sol = solve_ivp(self.system.f_ODE, tspan, current_state.numpy(), args=[current_input.detach().numpy()], events=self.guardsCheck)
        
        if sol.y.shape != (self.system.state_size,1): # Bug fix
            w = self.system.B([sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], sol.y[3][-1]],current_input.detach().numpy())
            next_state = [[sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], sol.y[3][-1], w]]
        else:
            w = self.system.B([sol.y[0][0], sol.y[1][0], sol.y[2][0], sol.y[3][0]],current_input.detach().numpy())
            next_state = [[sol.y[0][0], sol.y[1][0], sol.y[2][0], sol.y[3][0], w]]

        # Check if in intermediate timesteps in tspan constraints are violated
        for i in range(len(sol.y[0])):
            _ = self.system.B([sol.y[0][i], sol.y[1][i], sol.y[2][i], sol.y[3][i]],current_input.detach().numpy())

        if sol.t_events[0].size:

            if type(next_state) == torch.Tensor:
                next_state = next_state.numpy()

            impact_states = np.append(impact_states,next_state, axis=0)
            next_state = self.system.Rm[self.mode](torch.Tensor(sol.y_events[0][0]),current_input)
            reset_states = np.append(reset_states,[next_state.numpy()], axis=0)
            dt1 = sol.t_events[0] - tspan[0]
            dt2 = tspan[1] - sol.t_events[0]
            impact_mode_vec = np.append(impact_mode_vec, self.mode)

            tspan = np.linspace(sol.t_events[0], tspan[1], num=2)
            
            if self.mode == 0:
                self.mode = 1
            else:
                self.mode = 0

            impact_idx_vec = np.append(impact_idx_vec, iter)
            reset_mode_vec = np.append(reset_mode_vec, self.mode)
            transition_inputs = np.append(transition_inputs,[current_input.detach().numpy()], axis=0)
            hybrid_transitions += 1

            sol = solve_ivp(self.system.f_ODE, tspan, next_state.numpy(), args=[current_input.detach().numpy()], events=self.guardsCheck)
            
            w = self.system.B([sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], sol.y[3][-1]],current_input.detach().numpy())
            next_state = [[sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], sol.y[3][-1], w]]

            # Check if in intermediate timesteps in tspan constraints are violated
            for i in range(len(sol.y[0])):
                _ = self.system.B([sol.y[0][i], sol.y[1][i], sol.y[2][i], sol.y[3][i]],current_input.detach().numpy())

        storage = {"next_state":torch.Tensor(next_state), 
                   "next_mode":self.mode, 
                   "impact_states": impact_states, 
                   "reset_states": reset_states, 
                   "impact_mode_vec": impact_mode_vec, 
                   "reset_mode_vec": reset_mode_vec, 
                   "impact_idx_vec": impact_idx_vec, 
                   "transition_inputs": transition_inputs, 
                   "hybrid_transitions": hybrid_transitions}
        
        return storage
    
    def get_final_cost(self):
        J_out = 0
        goal = self.system.goal[:self.system.state_size-1]
        for i in range(self.horizon):
            x = self.x_trajectories[i,:self.system.state_size-1]
            u = self.u_trajectories[i,:]
            er = x - goal
            Q = self.system.Q[:self.system.state_size-1,:self.system.state_size-1]
            J_out += 0.5*er.T@Q@er + 0.5*u@self.system.R@u

        x = self.x_trajectories[-1,:self.system.state_size-1]
        er = x - goal
        Qt = self.system.Qt[:self.system.state_size-1,:self.system.state_size-1]
        J_out += 0.5*er.T@Qt@er
        return J_out