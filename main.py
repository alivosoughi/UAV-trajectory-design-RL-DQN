import numpy as np
import pandas as pd
import random
import argparse
import os
from environment.uav_env import Environment
from uav import Agent

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

class Designer:
	def __init__(self, args):
		current_path = os.path.dirname(__file__)
		self.env = Environment(args, current_path)
		self.episodes_number = args['episode_number']
		self.num_IotDevices = args['iot_devices_number']
		self.timeslot_num = args['timeslot_num']
		self.test = args['test']
		self.replay_steps = args['replay_steps']
		self.plot_frq = args['plot_frq']


	def simulate(self, agent):
		all_avg_min_thrpts = [0.0]
		all_trajectory = []
		max_score = -10000

		for episode_num in range(self.episodes_number):
			state, iot_devices_pos = self.env.reset()

			reward_all = 0
			time_step = 0

			cur_thrpts = np.zeros(self.num_IotDevices)
			min_thrpts = np.zeros(self.num_IotDevices)

			for time_step in range(self.timeslot_num):

				action = agent.greedy_actor(state)

				next_state, reward, cur_thrpts, min_thrpts, uav_trajectory = self.env.step(action, cur_thrpts, 
																							min_thrpts, 
																							all_avg_min_thrpts[-1], time_step)

				if not self.test:                	
					agent.observe((state.ravel(), action, reward, next_state.ravel()))
					agent.decay_epsilon()
					if time_step % self.replay_steps == 0:
						agent.replay()
					agent.update_target_model()

				state = next_state
				reward_all += reward

			all_avg_min_thrpts.append(np.average(min_thrpts))
			uav_trajectory = np.array(uav_trajectory)
			all_trajectory.append(uav_trajectory)

			if episode_num % self.plot_frq == 0:
				fig = plt.figure()
				ax = fig.gca(projection='3d')

				ax.scatter(iot_devices_pos[:, 0], iot_devices_pos[:, 1], iot_devices_pos[:, 2], c='r', marker='o')
				ax.plot(uav_trajectory[:, 0], uav_trajectory[:, 1], uav_trajectory[:, 2])
				
				ax.set_xlabel('X(meter)')
				ax.set_ylabel('Y(meter)')
				ax.set_zlabel('Z(meter)')

				plt.title('UAV trajectory')

				plt.show()

			print("Episode {p}, Average Throughput: {s}".format(p=episode_num, s=np.average(min_thrpts)))

			if not self.test:
				if episode_num % 100 == 0:
					if reward_all > max_score:
						agent.brain.save_model()
						max_score = reward_all




ARG_LIST = ['learning_rate', 'optimizer', 'memory_capacity', 'batch_size', 'target_model_update',
 			'maximum_exploration', 'timeslot_num', 'replay_steps', 'number_nodes', 'is_double_DQN', 
 			'memory', 'prioritization_scale', 'dueling']

def get_name_brain(args):
    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])
    return './models/weights_files/' + file_name_str + '_' + '.h5'


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('-e', '--episode-number', default=7_000, type=int, help='Number of episodes')
	parser.add_argument('-l', '--learning-rate', default=0.0001, type=float, help='Learning rate')
	parser.add_argument('-op', '--optimizer', choices=['Adam', 'RMSProp', 'SGD', 'Adadelta', 'Adagrad',
						'Adamax', 'Nadam', 'Ftrl'], default='Adam', help='Optimization method')
	parser.add_argument('-m', '--memory-capacity', default=50_000, type=int, help='Memory capacity')
	parser.add_argument('-b', '--batch-size', default=256, type=int, help='Batch size')
	parser.add_argument('-t', '--target-model-update', default=100, type=int,
                        help='update target model in every N episodes')
	parser.add_argument('-x', '--maximum-exploration', default=2_000, type=int, help='Maximum exploration step')
	parser.add_argument('-rs', '--replay-steps', default=4, type=float, help='replay and train memory in every N steps')
	parser.add_argument('-nn', '--number-nodes', default=256, type=int, help='Number of nodes in each layer of NN')
	parser.add_argument('-dd', '--is-double-DQN', action='store_false', default='use double DQN or normal DQN')
	parser.add_argument('-du', '--dueling', action='store_false', help='Enable Dueling architecture')
	parser.add_argument('-mt', '--memory', choices=['UER', 'PER'], default='PER', 
						help='method of drawing batches from memory')
	parser.add_argument('-pl', '--prioritization-scale', default=0.5, type=float, help='Scale for prioritization')
	parser.add_argument('-test', '--test', action='store_true', help='Enable the test phase')
	parser.add_argument('-in', '--iot-devices-number', default=12, type=int, help='The number of IOT devices')
	parser.add_argument('-ts', '--timeslot-num', default=100, type=int, help='number of time slots')
	parser.add_argument('-len', '--length', default=50, type=int, help='environment length')
	parser.add_argument('-wth', '--width', default=50, type=int, help='environment width')
	parser.add_argument('-hgt', '--height', default=25, type=int, help='environment height')
	parser.add_argument('-pe', '--plot-frq', default=5, type=int, help='plot trajectory every N episodes')
	parser.add_argument('-minh', '--min_height', default=10, type=int, help='uav minimum height')
	parser.add_argument('-maxh', '--max_height', default=20, type=int, help='uav maximum height')

	args = vars(parser.parse_args())

	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = '0' # specify which GPU(s) to be used

	designer = Designer(args)

	state_size = designer.env.state_size
	action_size = designer.env.action_size

	brain_file = get_name_brain(args)
	agent = Agent(state_size, action_size, brain_file, args)

	designer.simulate(agent)