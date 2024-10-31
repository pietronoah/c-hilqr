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
  
	def set_control_limits(self, limits):
		self.u_lim = limits
  
	def set_obstacles(self, obstacles):
		self.obstacle_info = obstacles

	def L(self, x, u):
		er = x - self.goal
		return 0.5*er.T@self.Q@er + 0.5*u.T@self.R@u 
	
	def Lt(self, x):
		er = x - self.goal
		return 0.5*er.T@self.Qt@er
	def L_prime(self, x,u):
		(l_x, l_u), ((l_xx, l_xu), (l_ux, l_uu)) = jacobian(self.L,(x,u)), hessian(self.L,(x,u))#, create_graph=True)
		l_x, l_u = l_x.squeeze(0), l_u.squeeze(0)
		return l_x, l_u, l_xx, l_ux, l_uu
	def Lt_prime(self, x):
		return (jacobian(self.Lt,x), hessian(self.Lt,x))
	
	def set_goal(self, x_goal):
		self.goal = x_goal

	def set_dt(self, d_t):
		self.dt = d_t

		
class BouncingBall2D_DBaS(DynamicalSystem):
	def __init__(self):
		super().__init__(5, 2) # Four states and two control input on the system
		self.dt = 1 # Default value
		self.goal = np.zeros(5) # 4 states + barrier state
		self.m = 1 # Ball's mass
		self.e = 0.75 # Coefficient of restitution
		self.init_mode = 0 # Initial mode 0 (falling)
		self.gravity = 9.81
		self.CV = 0

		# Store functions in a list for easier access later in the HiLQR interface
		self.Rm = [self.R12, self.R21]
		self.R_prime = [self.R12_prime, self.R21_prime]
		self.g = [self.g1, self.g2]
		self.g_prime = [self.g1_prime, self.g2_prime]
		self.salt = [self.salt1, self.salt2]
		self.g_ODE = [self.g1_ODE, self.g2_ODE]

		self.obstacle = []

	def B(self, x, u):
		px, py, vx, vy = x
		Fx, Fy = u
		B_out = 0
  
		# Check if obstacles are present or not
		if len(self.obstacle_info) == 0:
			return torch.tensor(0)

		for i in range(len(self.obstacle_info)):
			match self.obstacle_info[i][-1]:
				case 'circle':
					h = pow(px-self.obstacle_info[i][0],2) + pow(py-self.obstacle_info[i][1],2) - pow(self.obstacle_info[i][2],2) # >= 0
				case 'ellipse':
					h = pow(px-self.obstacle_info[i][0],2)/pow(self.obstacle_info[i][2],2) + pow(py-self.obstacle_info[i][1],2)/pow(self.obstacle_info[i][3],2) - 1 # >= 0
				case 'square':
					h = abs((px-self.obstacle_info[i][0]) + (py-self.obstacle_info[i][1])) + abs((px-self.obstacle_info[i][0]) - (py-self.obstacle_info[i][1])) - self.obstacle_info[i][2]  # >= 0

			if h > 0:
				B_out += 1/h
			else:
				B_out += torch.tensor(1e10) # Infinite cost inside of the circle
				self.CV = 1
    
		return B_out

	def h_circle(self, x):
		px, py, vx, vy, _ = x
		return pow(px-self.obstacle[0],2) + pow(py-self.obstacle[1],2) - pow(self.obstacle[2],2) # >= 0
	
	def h_ellipse(self, x):
		px, py, vx, vy, _ = x
		return pow(px-self.obstacle[0],2)/pow(self.obstacle[2],2) + pow(py-self.obstacle[1],2)/pow(self.obstacle[3],2) - 1 # >= 0
	
	def h_square(self, x):
		px, py, vx, vy, _ = x
		return abs((px-self.obstacle[0]) + (py-self.obstacle[1])) + abs((px-self.obstacle[0]) - (py-self.obstacle[1])) - self.obstacle[2]  # >= 0
		
    
	def dB_dx(self, x, u):
		px, py, vx, vy, _ = x
		Fx, Fy = u
		dB_dx_out = torch.zeros(1,self.state_size)

		# Check if obstacles are present or not
		if len(self.obstacle_info) == 0:
			return torch.zeros(1,self.state_size-1)[0,:]

		for i in range(len(self.obstacle_info)):
			self.obstacle = self.obstacle_info[i]
			match self.obstacle_info[i][-1]:
				case 'circle':
					h = self.h_circle(x)
					dhdx = jacobian(self.h_circle,x)
				case 'ellipse':
					h = self.h_ellipse(x)
					dhdx = jacobian(self.h_ellipse,x)
				case 'square':
					h = self.h_square(x)
					dhdx = jacobian(self.h_square,x)
				
			if h > 0:
				dBdh = (-1/pow(h.unsqueeze(0),2))
				dB_dx_out += torch.add(dB_dx_out,dBdh*dhdx)
			else:
				dB_dx_out += torch.tensor(1e10) # Infinite cost inside of the circle
				self.CV = 1

		return dB_dx_out[0,0:self.state_size-1]
		

	def f(self, x, u): # f1 and f2 are the same in this case
		px, py, vx, vy, B = x # Position and velocity of the system
		Fx, Fy = u # Trust

		ax = (Fx)/self.m # x acceleration
		ay = (Fy-self.m*self.gravity)/self.m # y acceleration

		w = self.B([px+vx*self.dt, py+vy*self.dt, vx+ax*self.dt, vy+ay*self.dt], u) #- self.B(self.goal[:self.state_size-1],u)

		return torch.cat(((px+vx*self.dt).unsqueeze(0), (py+vy*self.dt).unsqueeze(0), (vx+ax*self.dt).unsqueeze(0), (vy+ay*self.dt).unsqueeze(0), (w).unsqueeze(0)),0)

	def f_prime(self, x, u):
		return jacobian(self.f,(x,u))
	
	def f_cont(self, x, u): # f1 and f2 are the same in this case
		px, py, vx, vy, B = x # Position and velocity of the system
		Fx, Fy = u # Trust
		
		ax = (Fx)/self.m # x acceleration
		ay = (Fy-self.m*self.gravity)/self.m # y acceleration

		dBdx = self.dB_dx(x,u)
		wdot = dBdx@torch.tensor([vx, vy, ax, ay])

		return torch.cat(((vx).unsqueeze(0), (vy).unsqueeze(0), (ax).unsqueeze(0), (ay).unsqueeze(0), (wdot).unsqueeze(0)),0)
	
	def R12(self, x, u): 
		px, py, vx, vy, B = x # Position and velocity of the system
		return torch.cat(((px).unsqueeze(0), (py).unsqueeze(0), (vx).unsqueeze(0), (-self.e*vy).unsqueeze(0), (B).unsqueeze(0)),0)
	
	def R21(self, x, u): 
		px, py, vx, vy, B = x # Position and velocity of the system
		return torch.cat(((px).unsqueeze(0), (py).unsqueeze(0), (vx).unsqueeze(0), (vy).unsqueeze(0), (B).unsqueeze(0)),0)
	
	def R12_prime(self, x, u):
		return jacobian(self.R12,(x,u))
	
	def R21_prime(self, x, u):
		return jacobian(self.R21,(x,u))
	
	def g1(self,x,u):
		px, py, vx, vy, B = x # Position and velocity of the system
		return (-py).unsqueeze(0)
	
	def g2(self,x,u):
		px, py, vx, vy, B = x # Position and velocity of the system
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
		px, py, vx, vy, _ = x # Position and velocity of the system
		Fx, Fy = u # Trust
		
		ax = (Fx)/self.m # x acceleration
		ay = (Fy - self.m*self.gravity)/self.m # y acceleration

		wdot = 0
		
		return [(vx), (vy), (ax), (ay), (wdot)]

	def g1_ODE(t,x,u):
		px, py, vx, vy, B = x # Position and velocity of the system
		return (-py)

	def g2_ODE(t,x,u):
		px, py, vx, vy, B = x # Position and velocity of the system
		return (-vy)
	

