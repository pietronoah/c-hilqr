from os import X_OK
import numpy as np
from torch.autograd.functional import jacobian, hessian
from torch.autograd import grad 
import torch
from scipy.integrate import solve_ivp

class DynamicalSystem:
	# HiLQR cost terms
	def __init__(self, state_size, control_size):
		self.state_size = state_size
		self.control_size = control_size
	def set_cost(self, Q, R):
		self.Q = Q
		self.R = R
	def set_final_cost(self, Qt):
		self.Qt = Qt

	def set_horizon(self, H):
		self.horizon = H
  
	def set_control_limits(self, limits):
		self.u_lim = limits
  
	def set_obstacles(self, obstacles):
		self.obstacle_info = obstacles
		self.n_constraints = len(obstacles) + 2*self.control_size # Obstacles and upper/lower limit for each control input

	def L(self, x, u):
		er = x - self.goal
		return 0.5*er.T@self.Q@er + 0.5*u.T@self.R@u
	
	def Lt(self, x):
		er = x - self.goal
		return 0.5*er.T@self.Qt@er
	
	def L_prime(self, x,u):
		(l_x, l_u), ((l_xx, l_xu), (l_ux, l_uu)) = jacobian(self.L,(x,u)), hessian(self.L,(x,u))
		l_x, l_u = l_x.squeeze(0), l_u.squeeze(0)
		return l_x, l_u, l_xx, l_ux, l_uu
	
	def Lt_prime(self, x):
		return (jacobian(self.Lt,x), hessian(self.Lt,x))
	
	def set_goal(self, x_goal):
		self.goal = x_goal

	def set_dt(self, d_t):
		self.dt = d_t

		
class BouncingBall2D_AL(DynamicalSystem):
	def __init__(self):
		super().__init__(4, 2) # Four states and two control input on the system
		self.dt = 1 # Default value
		self.goal = np.zeros(4) # 4 states + barrier state
		self.m = 1 # Ball's mass
		self.e = 0.75 # Coefficient of restitution
		self.init_mode = 0 # Initial mode 0 (falling)
		self.gravity = 9.81
		self.CV = 0
		self.horizon = 100

		# Store functions in a list for easier access later in the HiLQR interface
		self.Rm = [self.R12, self.R21]
		self.R_prime = [self.R12_prime, self.R21_prime]
		self.g = [self.g1, self.g2]
		self.g_prime = [self.g1_prime, self.g2_prime]
		self.salt = [self.salt1, self.salt2]
		self.g_ODE = [self.g1_ODE, self.g2_ODE]

		self.obstacle = []
		self.n_constraints = 0
		self.u_lim = []

	def h_circle(self, x, u):
		px, py, vx, vy = x
		return (-torch.pow(px-torch.tensor(self.obstacle[0]),2) - torch.pow(py-torch.tensor(self.obstacle[1]),2) + torch.pow(torch.tensor(self.obstacle[2]),2)).unsqueeze(0) # <= 0
	
	def h_ellipse(self, x, u):
		px, py, vx, vy = x
		return (-torch.pow(px-torch.tensor(self.obstacle[0]),2)/torch.pow(torch.tensor(self.obstacle[2]),2) - torch.pow(py-torch.tensor(self.obstacle[1]),2)/torch.pow(torch.tensor(self.obstacle[3]),2) + 1).unsqueeze(0) # <= 0
	
	def h_square(self, x, u):
		px, py, vx, vy = x
		return (-torch.abs((px-torch.tensor(self.obstacle[0])) + (py-torch.tensor(self.obstacle[1]))) - torch.abs((px-torch.tensor(self.obstacle[0])) - (py-torch.tensor(self.obstacle[1]))) + torch.tensor(self.obstacle[2])).unsqueeze(0)  # <= 0
	
	def u_constraint(self,x,u):
		u_low = [sublist[0] for sublist in self.u_lim]
		u_up = [sublist[1] for sublist in self.u_lim]
		
		u_low_c = torch.tensor(u_low) - u
		u_up_c = u - torch.tensor(u_up)
		c = torch.cat((u_low_c, u_up_c), dim=0)
		return c


	def constraints(self, x, u):
		px, py, vx, vy = x
		Fx, Fy = u

		c = torch.tensor([])

		# Check if obstacles are present or not
		if len(self.obstacle_info) == 0:
			return c

		for i in range(len(self.obstacle_info)):
			self.obstacle = self.obstacle_info[i]
			match self.obstacle_info[i][-1]:
				case 'circle':
					h = self.h_circle(x, u)
				case 'ellipse':
					h = self.h_ellipse(x, u)
				case 'square':
					h = self.h_square(x, u)
				
			if h >= 0:
				self.CV = 1

			c = torch.cat((c, h), dim=0)

		c = torch.cat((c,self.u_constraint(x,u)),dim=0)

		return c
	

	def c_p(self, x, u):
		px, py, vx, vy = x
		Fx, Fy = u

		# Check if obstacles are present or not
		if len(self.obstacle_info) == 0:
			torch.tensor([])

		cx = torch.zeros((len(self.obstacle_info), self.state_size))
		cu = torch.zeros((len(self.obstacle_info), self.control_size))

		for i in range(len(self.obstacle_info)):
			self.obstacle = self.obstacle_info[i]
			match self.obstacle_info[i][-1]:
				case 'circle':
					cx[i,:], cu[i,:] = jacobian(self.h_circle,(x,u))
				case 'ellipse':
					cx[i,:], cu[i,:] = jacobian(self.h_ellipse,(x,u))
				case 'square':
					cx[i,:], cu[i,:] = jacobian(self.h_square,(x,u))

		temp = torch.zeros((self.state_size))
		temp1 = torch.diag(temp)
		cx = torch.cat((cx,temp1),dim=0)
		temp = -1.0*torch.ones((self.control_size))
		temp1 = torch.diag(temp)
		cu = torch.cat((cu,temp1),dim=0)
		temp = torch.ones((self.control_size))
		temp1 = torch.diag(temp)
		cu = torch.cat((cu,temp1),dim=0)

		return cx, cu

	def f(self, x, u): 
		px, py, vx, vy = x # Position and velocity of the system
		Fx, Fy = u # Trust
		
		ax = (Fx)/self.m # x acceleration
		ay = (Fy-self.m*self.gravity)/self.m # y acceleration

		return torch.cat(((px+vx*self.dt).unsqueeze(0), (py+vy*self.dt).unsqueeze(0), (vx+ax*self.dt).unsqueeze(0), (vy+ay*self.dt).unsqueeze(0)),0)
	
	def f_prime(self, x, u):
		return jacobian(self.f,(x,u))
	
	def f_cont(self, x, u): 
		px, py, vx, vy = x # Position and velocity of the system
		Fx, Fy = u # Trust
		
		ax = (Fx)/self.m # x acceleration
		ay = (Fy-self.m*self.gravity)/self.m # y acceleration 

		return torch.cat(((vx).unsqueeze(0), (vy).unsqueeze(0), (ax).unsqueeze(0), (ay).unsqueeze(0)),0)
	
	def R12(self, x, u): 
		px, py, vx, vy = x # Position and velocity of the system
		return torch.cat(((px).unsqueeze(0), (py).unsqueeze(0), (vx).unsqueeze(0), (-self.e*vy).unsqueeze(0)),0)
	
	def R21(self, x, u): 
		px, py, vx, vy = x # Position and velocity of the system
		return torch.cat(((px).unsqueeze(0), (py).unsqueeze(0), (vx).unsqueeze(0), (vy).unsqueeze(0)),0)
	
	def R12_prime(self, x, u):
		return jacobian(self.R12,(x,u))
	
	def R21_prime(self, x, u):
		return jacobian(self.R21,(x,u))
	
	def g1(self,x,u):
		px, py, vx, vy = x # Position and velocity of the system
		return (-py).unsqueeze(0)
	
	def g2(self,x,u):
		px, py, vx, vy = x # Position and velocity of the system
		return (-vy).unsqueeze(0)
	
	def g1_prime(self, x, u):
		return jacobian(self.g1,(x,u))
	
	def g2_prime(self, x, u):
		return jacobian(self.g2,(x,u))
	
	def salt1(self,x,u):
		F2 = self.f_cont(self.R12(x,u),u).reshape(1,-1).t()
		DxR12,_ = self.R12_prime(x,u)
		DxG12,_ = self.g1_prime(x,u)
		f_t = self.f_cont(x,u).reshape(1,-1).t()
		num = F2 - torch.matmul(DxR12,f_t)
		den = 0 + torch.matmul(DxG12,self.f_cont(x,u)) # 0 time derivative of the guard function

		return DxR12 + (num)*DxG12/(den) 
	
	def salt2(self,x,u):
		DxR21,_ = self.R21_prime(x,u)
		return DxR21 # 0 time derivative of the guard function
	
	# Continuos dynamic and guard functions definition for numerical integration with event detection
	def f_ODE(self, t, x, u): 
		px, py, vx, vy = x # Position and velocity of the system
		Fx, Fy = u # Trust
		
		ax = (Fx)/self.m # x acceleration
		ay = (Fy - self.m*self.gravity)/self.m # y acceleration
		
		return [(vx), (vy), (ax), (ay)]

	def g1_ODE(t,x,u):
		px, py, vx, vy = x # Position and velocity of the system
		return (-py)

	def g2_ODE(t,x,u):
		px, py, vx, vy = x # Position and velocity of the system
		return (-vy)
	

