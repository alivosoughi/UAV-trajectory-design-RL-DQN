import numpy as np
import random

from nn import Brain
from uniform_experience_replay import Memory as UER
from prioritized_experience_replay import Memory as PER

MAX_EPSILON = 1.0
MIN_EPSILON = 0.01

MIN_BETA = 0.4
MAX_BETA = 1.0

GAMMA = 0.95

class Agent(object):
	epsilon = MAX_EPSILON
	beta = MAX_BETA
	gamma = GAMMA

	def __init__(self, state_size, action_size, brain_name, args):
		self.state_size = state_size
		self.action_size = action_size
		learning_rate = args['learning_rate']
		self.brain = Brain(self.state_size, self.action_size, brain_name, args)
		self.memory_type = args['memory']

		if self.memory_type == 'UER':
			self.memory = UER(args['memory_capacity'])
		elif self.memory_type == 'PER':
			self.memory = PER(args['memory_capacity'], args['prioritization_scale'])
		else:
			print('Invalid memory model!')

		self.is_double_DQN = args['is_double_DQN']
		self.target_update_period = args['target_model_update']
		self.max_exploration_step = args['maximum_exploration']
		self.batch_size = args['batch_size']
		self.step = 0
		self.test = args['test']
		if self.test:
			self.epsilon = MIN_EPSILON


	def greedy_actor(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		else:
			return np.argmax(self.brain.predict_one_sample(state))


	def find_targets_per(self, batch):
		
		# every sample in PER batches has a form of: [sample_batch, sample_batch_indices, sample_batch_priorities]
		# every batch's form is: (state, action, reward, next_state)
		
		batch_size = len(batch)

		states = np.array([observation[1][0] for observation in batch])
		new_states = np.array([observation[1][3] for observation in batch])

		prediction = self.brain.predict(states)
		new_prediction = self.brain.predict(new_states)
		new_prediction_target = self.brain.predict(new_states, target=True)

		x = np.zeros((batch_size, self.state_size))
		y = np.zeros((batch_size, self.action_size))
		errors = np.zeros(batch_size)

		for i in range(batch_size):
			observation = batch[i][1]
			state = observation[0]
			action = observation[1]
			reward = observation[2]
			new_state = observation[3]

			actions_probability = prediction[i]
			old_value = actions_probability[action]

			if self.is_double_DQN: # if archtect was Double DQN
				actions_probability[action] = reward + self.gamma * new_prediction_target[i][np.argmax(new_prediction[i])]
			else:
				actions_probability[action] = reward + self.gamma * np.amax(new_prediction_target[i])

			x[i] = state
			y[i] = actions_probability
			errors[i] = np.abs(actions_probability[action] - old_value)

		return [x, y, errors]

	
	def find_targets_uer(self, batch):
		batch_size = len(batch)

		states = np.array([observation[0] for observation in batch])
		new_states = np.array([observation[3] for observation in batch])

		prediction = self.brain.predict(states)
		new_prediction = self.brain.predict(new_states)
		new_prediction_target = self.brain.predict(new_states, target=True)

		x = np.zeros((batch_size, self.state_size))
		y = np.zeros((batch_size, self.action_size))

		for i in range(batch_size):
			observation = batch[i]
			state = observation[0]
			reward = observation[1]
			reward = observation[2]
			new_state = observation[3]

			actions_probability = prediction[i]
			
			if self.is_double_DQN == 'DDQN':
				actions_probability[action] = reward + self.gamma * new_prediction_target[i][np.argmax(new_prediction[i])]
			elif self.is_double_DQN == 'DQN':
				actions_probability[action] = reward + self.gamma * np.amax(new_prediction_target[i])
			else:
				print('Invalid type for target network!')

			x[i] = state
			y[i] = actions_probability

		return [x, y]


	def observe(self, sample):

		if self.memory_type == 'UER':
			self.memory.remember(sample)

		elif self.memory_type == 'PER':
			_, _, errors = self.find_targets_per([[0, sample]])
			self.memory.remember(sample, errors[0])

		else:
			print('Invalid memory model!')


	def decay_epsilon(self):
		# slowly decrease Epsilon based on our experience
		self.step += 1

		if self.test:
			self.epsilon = MIN_EPSILON
			self.beta = MAX_BETA
		
		else:
			if self.step < self.max_exploration_step:
				self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * (self.max_exploration_step - self.step)/self.max_exploration_step
				self.beta = MAX_BETA + (MIN_BETA - MAX_BETA) * (self.max_exploration_step - self.step)/self.max_exploration_step
			
			else:
				self.epsilon = MIN_EPSILON


	def replay(self):

		if self.memory_type == 'UER':
			batch = self.memory_type.sample(self.batch_size)
			x, y = self.find_targets_uer(batch)
			self.brain.train(x, y)

		elif self.memory_type == 'PER':
			[batch, batch_indices, batch_priorities] = self.memory.sample(self.batch_size)
			x, y, errors = self.find_targets_per(batch)

			normalized_batch_priorities = [float(i) / sum(batch_priorities) for i in batch_priorities]
			importance_sampling_weights = [(self.batch_size * i) ** (-1 * self.beta)
										   for i in normalized_batch_priorities]
			normalized_importance_sampling_weights = [float(i) / max(importance_sampling_weights)
													  for i in importance_sampling_weights]
			sample_weights = [errors[i] * normalized_importance_sampling_weights[i] for i in range(len(errors))]

			self.brain.train(x, y, np.array(sample_weights))

			self.memory.update(batch_indices, errors)

		else:
			print('Invalid memory model!')


	def update_target_model(self):
		if self.step % self.target_update_period == 0:
			self.brain.update_target_model()