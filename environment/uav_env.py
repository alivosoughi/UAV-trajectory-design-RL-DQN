import random
import numpy as np
import sys
import os

STATE_SPACE_SIZE = 27
MOVES = [-1, 0, 1]

class Environment:
	A = [i for i in range(0, STATE_SPACE_SIZE)]
	A_DIFF = [(i, j, k) for i in MOVES for j in MOVES for k in MOVES]

	def __init__(self, args, current_path):
		self.length = args['length']
		self.width = args['width']
		self.height = args['height']
		self.min_height = args['min_height']
		self.max_height = args['max_height']
		self.timeslot_num = args['timeslot_num']
		self.num_IotDevices = args['iot_devices_number']
		self.state_size = 3 + self.num_IotDevices * 2
		self.action_size = STATE_SPACE_SIZE
		self.uav_pos = []
		self.iot_devices_pos = []
		self.uav_trajectory = []
		self.space_grid = []
		self.positions_idx = []
		self.a = []
		self.R = []


	def set_positions_idx(self):

		space_grid = [(i, j, k) for i in range(0, self.length) 
							 for j in range(0, self.width)
							 for k in range(0, self.height)]

		space_grid = np.array(space_grid)

		space2D_state_size = self.length * self.width
		positions_idx = np.random.choice(space2D_state_size, size=self.num_IotDevices+1, replace=False) * self.height

		return [space_grid, positions_idx]


	def reset(self):

		[self.space_grid, positions_idx] = self.set_positions_idx()

		self.uav_pos = self.space_grid[positions_idx[0], :]
		self.iot_devices_pos = self.space_grid[positions_idx[1:], :]

		self.uav_trajectory.append(self.uav_pos)

		self.a = np.zeros(self.num_IotDevices)

		self.R = np.zeros(self.num_IotDevices)

		initial_state = np.concatenate((self.uav_pos, self.a, self.R), axis=0)
		
		return initial_state, self.iot_devices_pos


	def step(self, agent_action, cur_thrpts, min_thrpts, last_avg_min_thrpts, time_step):

		reward = 0

		# update the position of agents
		uav_pos_temp = self.update_positions(self.uav_pos, agent_action)

		dims = np.array([self.length, self.width, self.height])
		for i in range(0, 3):
					if uav_pos_temp[i] < 0:
						uav_pos_temp[i] = 0
						reward -= 1
					if uav_pos_temp[i] >= dims[i] - 1:
						uav_pos_temp[i] = dims[i] - 1
						reward -= 1       

		if uav_pos_temp[2] < self.min_height:
			reward -= 1
		if uav_pos_temp[2] > self.max_height:
			reward -= 1

		self.uav_trajectory.append(uav_pos_temp)
		self.uav_pos = uav_pos_temp

		uav_iot_dists = [np.linalg.norm(dev_pos - self.uav_pos, 1) for dev_pos in self.iot_devices_pos]

		device_t = np.argmin(uav_iot_dists)
	   
		self.a[device_t] += 1

		device_t_thrpt = self.cal_thrpt(self.iot_devices_pos[device_t], self.uav_pos)

		self.R[device_t] += device_t_thrpt

		if cur_thrpts[device_t] == 0:
			min_thrpts[device_t] = device_t_thrpt
		elif cur_thrpts[device_t] <= device_t_thrpt:
			min_thrpts[device_t] = device_t_thrpt
			reward -= 1

		cur_thrpts[device_t] = device_t_thrpt

		new_state = np.concatenate((self.uav_pos, self.a, self.R), axis=0)
		
		if time_step == self.timeslot_num-1:
			if np.min(min_thrpts) == 0.0:
				reward -= 2

			if last_avg_min_thrpts > np.average(min_thrpts):
				reward -= 1

			return [new_state, reward, cur_thrpts, min_thrpts, self.uav_trajectory]

		return [new_state, reward, cur_thrpts, min_thrpts, []]


	def cal_thrpt(self, dev_pos, uav_pos):
		d = np.linalg.norm(dev_pos - uav_pos, 1)
		return np.log2(1+1/(d+0.001))


	def update_positions(self, pos, act):
		move = self.A_DIFF[act]
		final_positions = pos + act
		return final_positions


