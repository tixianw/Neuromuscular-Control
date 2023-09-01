import numpy as np
from tqdm import tqdm
from elastica import *
from actuation_muscles import *
from tools import _aver, _aver_kernel, _diff, _diff_kernel, relu, RK_solver, gaussian, sech

class Cable:
	def __init__(self, ds, dt, tau, lmd, V_rest, V0, tau_adapt, inhibition_weight, adaptation_weight):
		self.ds = ds
		self.dt = dt
		self.tau = tau
		self.lmd = lmd
		self.V_rest = V_rest
		self.V0 = V0 # V_rest
		self.V = self.V0.copy()
		### inhibition + adaptation
		self.tau_adapt = tau_adapt
		self.inhibition_weight = inhibition_weight
		self.adaptation_weight = adaptation_weight
		self.V_adapt = np.zeros_like(self.V0)
	
	def dynamics_cable(self, V, I):
		Vss = _diff_kernel(_diff(V)/self.ds)/self.ds
		### Inhibition
		inhibition = V.copy() # Vss.copy() # 
		inhibition[[0,1]] = V[[1,0]] # Vss[[1,0]] # 
		inhibition[-1] *= 0
		dVdt = self.lmd*self.lmd * Vss - V + self.V_rest + I - self.V_adapt - self.inhibition_weight * relu(inhibition)
		### Boundary conditions
		# dVdt[..., 0] *= 0 # Dirichlet cond. at base
		# dVdt[..., -1] = dVdt[..., -2] # Neumann cond. at tip
		# dVdt[..., -1] *= 0 # Dirichlet cond. at tip
		return dVdt / self.tau

	def dynamics_adapt(self, V_adapt, I=None):
		dVdt_adapt = -V_adapt + self.adaptation_weight * relu(self.V)
		dVdt_adapt[-1] *= 0
		return dVdt_adapt / self.tau_adapt

class NeuralCtrl:
	def __init__(self, env, callback_list: dict, step_skip: int, ramp_up_time=0.0):
		self.env = env
		self.callback_list = callback_list
		self.every = step_skip
		assert ramp_up_time >= 0.0
		if ramp_up_time == 0:
			self.ramp_up_time = 1e-14
		else:
			self.ramp_up_time = ramp_up_time
		self.count = 0

		self.n_elem = env.arm_param['n_elem']
		L = env.arm_param['L']
		# radius = env.arm_param['radius']
		ds = L / self.n_elem
		self.s = np.linspace(0, L, self.n_elem+1)
		dt = env.time_step             
		tau = 0.04 # 0.04 #
		lmd0 = 0.1 # -0.1
		lmd = lmd0 # abs(lmd0) * np.sqrt(radius/radius[0]) # 
		tau_adapt = tau * 10 # 2
		inhibition_weight = 0. # 2. # 0.2 #
		adaptation_weight = 1. # 2. # 2. # 
		### initial voltage
		self.V_rest = 0. # -50
		V_t0 = 60 # 40 # 50 # 
		V_b0 = 40 # 50 # self.V_rest #
		V_t1 = 80 # 120 # 75
		V_b1 = 0 # -25 # V_b0
		if env.flag_shooting:
			V0 = env.init_data['rest_V']
		else:
			V0 = np.vstack([
				np.ones([1, self.n_elem+1]) * ((V_t1 - V_t0)/L*(self.s) + V_t0), 
				np.ones([1, self.n_elem+1]) * ((V_b1 - V_b0)/L*(self.s) + V_b0), 
				np.ones([1, self.n_elem+1]) * self.V_rest,
				])
		self.neuron_param = {
			'tau': tau, 
			'lmd': lmd0, # negative lambda means tapered
			'tau_adapt': tau_adapt,
			'inhibition': inhibition_weight,
			'adaptation': adaptation_weight,
			'V0': [V_t0, V_b0, V_t1, V_b1],
		}
		print('tau',tau,'lmd',lmd0,'tauA',tau_adapt,'inhibit',inhibition_weight,'adapt',adaptation_weight,'V',[V_t0, V_b0, V_t1, V_b1])
		
		### saturation function
		self.gap = 0.01
		V_ub = 80. # 0
		self.mean = 0.5 * (self.V_rest + V_ub)
		self.var = 2 / (self.V_rest + V_ub) * np.arctanh(2*self.gap - 1)
		if self.mean > 0:
			self.var *= -1

		## initialize cable equation
		self.neuron = Cable(ds, dt, tau, lmd, self.V_rest, V0, tau_adapt, inhibition_weight, adaptation_weight)
		self.ctrl_mag = np.zeros([3, self.n_elem+1])
		self.I = np.zeros_like(self.ctrl_mag)

		### PID controller
		self.error_sum = np.zeros(self.n_elem-1)
		self.error = np.zeros(self.n_elem-1)
		self.delta_error = np.zeros(self.n_elem-1)

		### Sensory Feedback controller
		self.mu = 0
	
	def backstepping(self, u):
		### first
		# gamma = 100
		# self.I = self.neuron.tau * gamma * (self.u_to_V(u) - self.neuron.V) + self.neuron.V - \
		# 	self.neuron.lmd*self.neuron.lmd * \
		# 		_diff_kernel(_diff(self.neuron.V)/self.neuron.ds)/self.neuron.ds
		### second
		# self.gamma = 1 / self.neuron.tau * 1000
		# inhibition = self.neuron.V.copy()
		# inhibition[[0,1]] = self.neuron.V[[1,0]]
		# inhibition[-1] *= 0
		# self.I = self.neuron.tau * self.gamma * (self.u_to_V(u) - self.neuron.V) + self.neuron.V + \
		# 	self.neuron.V_adapt + self.neuron.inhibition_weight * relu(inhibition) - \
		# 	self.neuron.lmd*self.neuron.lmd * _diff_kernel(_diff(self.neuron.V)/self.neuron.ds)/self.neuron.ds
		### third
		# self.gamma = 1 # 10
		# self.I = self.gamma * (self.u_to_V(u) - self.neuron.V) + self.neuron.V + self.neuron.adaptation_weight * relu(self.neuron.V) - \
		# 	self.neuron.lmd*self.neuron.lmd * _diff_kernel(_diff(self.neuron.V)/self.neuron.ds)/self.neuron.ds
		### fourth
		self.gamma = 3
		self.I = self.gamma * (self.u_to_V(u) - self.neuron.V) + self.neuron.V + self.neuron.adaptation_weight * relu(self.neuron.V) - \
			self.neuron.lmd*self.neuron.lmd * _diff_kernel(_diff(self.u_to_V(u))/self.neuron.ds)/self.neuron.ds
	
	def LM_choice(self, array):
		return np.where(array>=0), np.where(array<0)
	
	def PID(self, system, kappa):
		order = 0 # 0.5 # 1
		KP = 6 # 40 # 1 # 
		KI = 10 # 15 # 0.01 #
		KD = 0 ## (* dt) # 0.06 # 50 # 1e-4 #
		self.PID_param = [KP, KI, KD, order]
		self.error_old = self.error.copy()
		self.error = kappa - (-system.kappa[0,:])
		self.error_sum += self.error
		self.delta_error = self.error - self.error_old

		input = KP * self.error + KI * self.error_sum * self.neuron.dt + KD * self.delta_error # / self.neuron.dt
		input *= _aver((system.radius/system.radius[0])**order)
		idx_top, idx_bottom = self.LM_choice(input)
		self.I[0, idx_top] = input[idx_top]
		self.I[1, idx_bottom] = -input[idx_bottom]
	
	def sensoryfeedback(self, system, u):
		self.mu = 500 # 250 # 
		self.I = self.mu * u
	
	def get_I(self, time, system, desired_curvature, desired_activation):
		# self.I = np.zeros([3, self.n_elem+1])
		# self.backstepping(desired_activation)
		# self.PID(system, desired_curvature)
		self.sensoryfeedback(system, desired_activation)

	def cable_eq(self, system):
		self.neuron.V = RK_solver(self.neuron.dynamics_cable, self.neuron.V, self.neuron.dt, self.I)
		self.neuron.V_adapt = RK_solver(self.neuron.dynamics_adapt, self.neuron.V_adapt, self.neuron.dt, self.I)

	def u_to_V(self, u):
		u = np.clip(u, self.gap, 1-self.gap)
		array = self.mean + 1/self.var * np.arctanh(2*u - 1)
		return array - self.V_rest
	
	def V_to_u(self, time):
		factor = min(1.0, time / self.ramp_up_time)
		self.ctrl_mag[:, :] = factor * (0.5 + 0.5 * np.tanh(self.var * (self.neuron.V - self.mean)))
		# self.ctrl_mag = np.clip(self.ctrl_mag, 0, 1)
	
	def v_to_u(self, V):
		return 0.5 + 0.5 * np.tanh(self.var * (V - self.mean))
	
	def neural_ctrl(self, time, system, desired_curvature, desired_activation):
		self.get_I(time, system, desired_curvature, desired_activation)
		self.callback()
		self.cable_eq(system)
		self.V_to_u(time)
		self.count += 1
		return self.ctrl_mag
	
	def callback(self):
		if self.count % self.every == 0:
			self.callback_list['I'].append(self.I.copy())
			self.callback_list['V'].append(self.neuron.V.copy())