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
from utils.utils_AL import * 

# Define the main class iLQR
class AL_HiLQR:
    def __init__(self, system, initial_state, u_init, horizon = 300):
        self.system = system
        self.horizon = horizon
        self.initial_state = initial_state
        self.x_trajectories = torch.zeros((self.horizon + 1, self.system.state_size))
        self.x_trajectories[0,:] = self.initial_state
        self.x_trajectories_history = torch.zeros((self.horizon + 1, self.system.state_size, 100))
        self.u_trajectories = u_init
        self.u_trajectories_history = torch.zeros((self.horizon, self.system.control_size, 100))
        self.x_trajectories_low_lr_history = torch.zeros((self.horizon+1, self.system.state_size, 100))
        self.u_trajectories_low_lr_history = torch.zeros((self.horizon, self.system.control_size, 100))
        self.ks = torch.zeros((self.horizon, self.system.control_size))
        self.Ks = torch.zeros((self.horizon, self.system.control_size, self.system.state_size))
        self.J_new = 10000
        self.J = 1000
        self.J_history = torch.zeros((100,))
        self.deltaV = 0
        self.alphas = 0.4**np.arange(20) # Line search candidates
        self.dalpha1 = 0
        self.dalpha2 = 0
        self.niter = 1
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

        # Lagrangian terms
        self.λ = torch.zeros((self.horizon + 1, self.system.n_constraints))
        self.μ = 0.1*torch.ones((self.horizon + 1, self.system.n_constraints))
        self.Iμ = torch.zeros((self.system.n_constraints, self.system.n_constraints, self.horizon + 1))
        self.φ = 10
        self.μ_max = 1e6

        self.c_viol = 0 # constraint violation
        self.c_viol_max = 1e-2 # constraint convergency
        self.outer_loop_iter = 1
        self.outer_loop_iter_max = 10

        self.unfeasible_start = 0

        self.k_iter = 0
        self.total_iter_HiLQR = 0

        self.backward_pass_eval = 0
        self.forward_pass_eval = 0

    def fit(self):
        self.system.CV = 0
        print_logo()

        start = time.time()
        
        self.rollout()
        self.rollout_x = self.x_trajectories

        if self.system.CV == 1:
            print()
            print("Unfeasible initial trajectory")
            print()
            self.unfeasible_start = 1
        
        # Update penalty value based on the rollout trajectory
        self.λ, self.μ, self.Iμ = self.update_λμIμ(self.x_trajectories, self.u_trajectories, self.λ, self.μ, self.Iμ)

        while self.outer_loop_iter < self.outer_loop_iter_max:
            print()
            print('Outer loop iteration n. ', self.outer_loop_iter)
            print()
            print_header()

            # Inner loop
            self.HiLQR()
            # Update penalty terms based on new trajectory
            self.λ, self.μ, self.Iμ = self.update_λμIμ(self.x_trajectories, self.u_trajectories, self.λ, self.μ, self.Iμ)

            if self.c_viol < self.c_viol_max:
                print()
                print('Constraint convergency reached! Optimal trajectory found!')
                print()
                end = time.time()

                end_error = np.linalg.norm(self.x_trajectories[-1,:2] - self.system.goal[:2])
                if end_error < 0.05:
                    self.convergency = 1
                else:
                    self.convergency = 0

                try:
                    print_summary(self.outer_loop_iter,self.total_iter_HiLQR,self.backward_pass_eval, self.forward_pass_eval,format_large_number(self.J),format_number(self.c_viol.item()),format_number(end-start))
                except:
                    print_summary(self.outer_loop_iter,self.total_iter_HiLQR,self.backward_pass_eval, self.forward_pass_eval,format_large_number(self.J),format_number(self.c_viol),format_number(end-start))
                return
            else:
                self.outer_loop_iter += 1
                print('Constraint violation: ', self.c_viol)

            # Repeat rollout to get the cost of the starting trajectory with the updated penalty terms
            self.rollout()

        if self.outer_loop_iter == self.outer_loop_iter_max:
            print()
            print('Outer loop max iter reached!')
            self.convergency = 0
            end = time.time()
            try:
                print_summary(self.outer_loop_iter,self.total_iter_HiLQR,self.backward_pass_eval, self.forward_pass_eval,format_large_number(self.J),format_number(self.c_viol.item()),format_number(end-start))
            except:
                print_summary(self.outer_loop_iter,self.total_iter_HiLQR,self.backward_pass_eval, self.forward_pass_eval,format_large_number(self.J),format_number(self.c_viol),format_number(end-start))
            return
        


    def HiLQR(self):
        self.niter = 1
        armijo_threshold = 0.1
        while self.concluded != 1 and self.niter < self.max_iter:
            
            self.backward_pass()
            CV_n = 0
            CV_percentage = 0
            it_alpha = 0
            armijo_flag = 0
            while(it_alpha < len(self.alphas) and armijo_flag == 0):
                alpha = self.alphas[it_alpha]
                storage = self.forward_pass(alpha)
                cost_diff = self.J - self.J_new
                expected_cost_redu = alpha*self.expected_cost_redu_grad_ + pow(alpha,2)*self.expected_cost_redu_hess_
                
                #armijo_flag = 1 if cost_diff/expected_cost_redu > armijo_threshold else 0
                #if(armijo_flag == 1):
                if cost_diff > 0:
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
                    if it_alpha > 0:
                        CV_percentage = CV_n/it_alpha
                    else:
                        CV_percentage = 0

                    self.total_iter_HiLQR += 1
                    
                    print_current_iteration(self.niter, format_large_number(self.J), format_large_number(-cost_diff.numpy()), format_large_number(end_error), format_number(self.alphas[it_alpha]), it_alpha+1, CV_percentage)
                else:
                    self.x_trajectories_low_lr_history[:,:,it_alpha] = storage["x_new_trajectories"]
                    self.u_trajectories_low_lr_history[:,:,it_alpha] = storage["u_new_trajectories"]
                    it_alpha += 1
                    if self.system.CV == 1:
                        CV_n += 1
                        self.system.CV = 0
    
            self.x_trajectories_history[:,:,self.total_iter_HiLQR-1] = self.x_trajectories
            self.u_trajectories_history[:,:,self.total_iter_HiLQR-1] = self.u_trajectories
            self.J_history[self.total_iter_HiLQR-1] = self.J

            if abs(cost_diff) < self.tolerance:
                print()
                print("Internal loop convergency reached!")
                break

            # If learning rate is low, then stop optimization
            if(alpha<=self.alphas[-1]):
                print()
                print("Stopping optimization, low alpha")
                self.low_lr = 1
                self.total_iter_HiLQR += 1
                break
            
            self.niter += 1

    
    def update_λμIμ(self,x_traj,u_traj,λ,μ,Iμ):
        temp_cv = 0.0

        if self.system.n_constraints == 0:
            return λ,μ,Iμ
        
        for i in range(self.horizon):
            const_eval = self.system.constraints(x_traj[i,:],u_traj[i,:]) # constraint <=0 if respected
            max_value, _ = torch.max(const_eval,dim=0)
            temp_cv = max(temp_cv,max_value.numpy()) 
            for j in range(self.system.n_constraints):
                λ[i,j] = torch.max(torch.tensor(0), λ[i,j] + μ[i,j]*const_eval[j]) # only inequality constraints
                μ[i,j] = self.φ * μ[i,j]
                if const_eval[j] < 0 and λ[i,j] == 0:
                    Iμ[j,j,i] = 0
                else:
                    Iμ[j,j,i] = min(μ[i,j],self.μ_max)

        self.c_viol = max(0,temp_cv)

        print('Penalty terms updated!')

        return λ, μ, Iμ
            

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

            self.k_iter = i
            self.J += self.augmented_lagrangian(self.x_trajectories[i,:], self.u_trajectories[i,:])

        self.impact_states_ = impact_states
        self.reset_states_ = reset_states
        self.impact_mode_vec_ = impact_mode_vec
        self.reset_mode_vec_ = reset_mode_vec
        self.impact_idx_vec_ = impact_idx_vec
        self.transition_inputs_ = transition_inputs
        self.hybrid_transitions_ = hybrid_transitions

        self.k_iter = self.horizon
        self.J += self.augmented_lagrangian(self.x_trajectories[-1,:], torch.zeros((1, self.system.control_size))[0])
        
        
    def augmented_lagrangian(self,x,u):
        inequality = self.system.constraints(x,u)
        inactive = ((inequality < 0.0) & (self.λ[self.k_iter,:] == 0))
        active_set = np.invert(inactive)
        if self.k_iter == self.horizon:
            return self.system.Lt(x) + self.λ[self.k_iter,:].T @ inequality + 0.5 * self.μ[self.k_iter,:] * inequality.T @ (active_set * inequality)
        else:
            return self.system.L(x,u) + self.λ[self.k_iter,:].T @ inequality + 0.5 * self.μ[self.k_iter,:] * inequality.T @ (active_set * inequality)
        

    def AL_L_prime(self, x,u):
        (l_x, l_u), ((l_xx, l_xu), (l_ux, l_uu)) = jacobian(self.augmented_lagrangian,(x,u)), hessian(self.augmented_lagrangian,(x,u))
        l_x, l_u = l_x.squeeze(0), l_u.squeeze(0)
        return l_x, l_u, l_xx, l_ux, l_uu
    

    def backward_pass(self):

        symmetrize = lambda x: (x + x.T) / 2  

        self.backward_pass_eval += 1
        self.delta_V = 0

        self.k_iter = self.horizon
        p, _, P, _, _ = self.AL_L_prime(self.x_trajectories[-1,:],torch.zeros((1, self.system.control_size))[0])

        c = self.system.constraints(self.x_trajectories[-1,:], torch.zeros((1, self.system.control_size))[0])
        c_x, c_u = self.system.c_p(self.x_trajectories[-1,:], torch.zeros((1, self.system.control_size))[0])

        p = p + c_x.T @ self.Iμ[:,:,-1] @ c + c_x.T @ self.λ[-1,:]
        P = P + c_x.T @ self.Iμ[:,:,-1] @ c_x
        
        self.dalpha1 = 0
        self.dalpha2 = 0

        expected_cost_redu = 0
        expected_cost_redu_grad = 0
        expected_cost_redu_hess = 0

        # Back propagate along the trajectory
        for i in range(self.horizon-1,-1,-1): 
            x = self.x_trajectories[i,:]
            u = self.u_trajectories[i,:]

            c = self.system.constraints(x, u)
            c_x, c_u = self.system.c_p(x,u)
            
            self.system.iter = i

            idx, = np.where(self.impact_idx_vec_ == i)

            if idx.size > 0:
                for j in range(idx.size-1,-1,-1):
                    impact_states = self.impact_states_[self.system.state_size*idx[j]:self.system.state_size*(idx[j]+1)]
                    impact_inputs = self.transition_inputs_[self.system.control_size*idx[j]:self.system.control_size*(idx[j]+1)]
                    impact_mode = self.impact_mode_vec_[idx[j]:(idx[j]+1)]
                    salt = self.system.salt[impact_mode[0]](torch.Tensor(impact_states),torch.Tensor(impact_inputs))
                    p = torch.matmul(torch.transpose(salt,0,1),p.reshape(1,-1).t()).squeeze().t()
                    P = torch.matmul(torch.matmul(torch.transpose(salt,0,1),P),salt)
                              
            f_x, f_u = self.system.f_prime(x, u)

            self.k_iter = i
            l_x, l_u, l_xx, l_ux, l_uu = self.AL_L_prime(x,u)

            Q_xx = l_xx + f_x.T@P@f_x + c_x.T@self.Iμ[:,:,i]@c_x
            Q_uu = l_uu + f_u.T@P@f_u + c_u.T@self.Iμ[:,:,i]@c_u
            Q_ux = l_ux + f_u.T@P@f_x + c_u.T@self.Iμ[:,:,i]@c_x
            Q_x = l_x + f_x.T@p + c_x.T@self.Iμ[:,:,i]@c + c_x.T@self.λ[i,:]
            Q_u = l_u + f_u.T@p + c_u.T@self.Iμ[:,:,i]@c + c_u.T@self.λ[i,:]

            self.ks[i] = -torch.linalg.inv(Q_uu)@Q_u
            self.Ks[i] = -torch.linalg.inv(Q_uu)@Q_ux

            P = symmetrize(Q_xx + self.Ks[i].T@Q_uu@self.Ks[i] + 2*self.Ks[i].T@Q_ux)
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
        self.forward_pass_eval += 1
        self.mode = self.system.init_mode

        x_new_trajectories = torch.zeros((self.horizon + 1, self.system.state_size))
        u_new_trajectories = torch.zeros((self.horizon, self.system.control_size))

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
 
            self.k_iter = i
            self.J_new += self.augmented_lagrangian(x_new_trajectories[i,:], u_new_trajectories[i,:])

        self.k_iter = self.horizon
        self.J_new += self.augmented_lagrangian(x_new_trajectories[-1,:], torch.zeros((1, self.system.control_size))[0])

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
            next_state = [[sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], sol.y[3][-1]]]
        else:
            next_state = [[sol.y[0][0], sol.y[1][0], sol.y[2][0], sol.y[3][0]]]

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
            
            next_state = [[sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], sol.y[3][-1]]]

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
        for i in range(self.horizon):
            J_out += self.system.L(self.x_trajectories[i,:], self.u_trajectories[i,:])
        J_out += self.system.Lt(self.x_trajectories[-1,:])
        return J_out