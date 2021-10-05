from keras.models import Model
from keras.layers import Dense, Lambda, Input, Concatenate
from keras.optimizers import *
import tensorflow as tf
from keras import backend
import os

HUBER_LOSS_DELTA = 1.0

def huber_loss(y_true, y_predict):
	err = y_true - y_predict

	cond = backend.abs(err) < HUBER_LOSS_DELTA
	L2 = 0.5 * backend.square(err)
	L1 = HUBER_LOSS_DELTA * (backend.abs(err) - 0.5 * HUBER_LOSS_DELTA)
	loss = tf.where(cond, L2, L1)

	return backend.mean(loss)

class Brain(object):
	def __init__(self, state_size, action_size, brain_name, args):
		self.state_size = state_size
		self.action_size = action_size
		self.weight_backup_dir = brain_name
		self.batch_size = args['batch_size']
		self.learning_rate = args['learning_rate']
		self.test = args['test']
		self.num_nodes = args['number_nodes']
		self.dueling = args['dueling']
		self.optimizer_model = args['optimizer']
		self.online_model = self._build_model()
		self.target_model = self._build_model()


	def _build_model(self):

		if self.dueling:
			x = Input(shape=(self.state_size,))

			# a series of fully connected layer for estimating V(s)

			y11 = Dense(self.num_nodes, activation='relu')(x)
			y12 = Dense(self.num_nodes, activation='relu')(y11)
			y13 = Dense(1, activation="linear")(y12)

			# a series of fully connected layer for estimating A(s,a)

			y21 = Dense(self.num_nodes, activation='relu')(x)
			y22 = Dense(self.num_nodes, activation='relu')(y21)
			y23 = Dense(self.action_size, activation="linear")(y22)

			w = Concatenate(axis=-1)([y13, y23])

			# combine V(s) and A(s,a) to get Q(s,a)
			z = Lambda(lambda a: backend.expand_dims(a[:, 0], axis=-1) + a[:, 1:] - backend.mean(a[:, 1:], keepdims=True),
					   output_shape=(self.action_size,))(w)

		else:
			x = Input(shape=(self.state_size,))

			# a series of fully connected layer for estimating Q(s,a)
			y1 = Dense(self.num_nodes, activation='relu')(x)
			y2 = Dense(self.num_nodes, activation='relu')(y1)
			z = Dense(self.action_size, activation="linear")(y2)

		model = Model(inputs=x, outputs=z)

		if self.optimizer_model == 'Adam':
			optimizer = Adam(lr=self.learning_rate, clipnorm=1.)
		elif self.optimizer_model == 'RMSProp':
			optimizer = RMSprop(lr=self.learning_rate, clipnorm=1.)
		elif self.optimizer_model == 'SGD':
			optimizer = SGD(lr=self.learning_rate, clipnorm=1.)
		elif self.optimizer_model == 'Adadelta':
			optimizer = Adadelta(lr=self.learning_rate, clipnorm=1.)
		elif self.optimizer_model == 'Adagrad':
			optimizer = Adagrad(lr=self.learning_rate, clipnorm=1.)
		elif self.optimizer_model == 'Adamax':
			optimizer = Adamax(lr=self.learning_rate, clipnorm=1.)
		elif self.optimizer_model == 'Nadam':
			optimizer = Nadam(lr=self.learning_rate, clipnorm=1.)
		elif self.optimizer_model == 'Ftrl':
			optimizer = Ftrl(lr=self.learning_rate, clipnorm=1.)
		else:
			print('Invalid optimizer!')

		model.compile(loss=huber_loss, optimizer=optimizer)
		
		if self.test:
			if not os.path.isfile(self.weight_backup_dir):
				print('Error:no file')
			else:
				model.load_weights(self.weight_backup_dir)

		return model


	def train(self, x, y, sample_weight=None, epochs=1, verbose=0): 
		# print("trained!!!")
		self.online_model.fit(x, y, batch_size=len(x), sample_weight=sample_weight, epochs=epochs, verbose=verbose)


	def predict(self, state, target=False):
		if target:  # get prediction from target network
			return self.target_model.predict(state)
		else:  # get prediction from local network
			return self.online_model.predict(state)


	def predict_one_sample(self, state, target=False):
		return self.predict(state.reshape(1,self.state_size), target=target).flatten()


	def update_target_model(self):
		self.target_model.set_weights(self.online_model.get_weights())


	def save_model(self):
		self.online_model.save(self.weight_backup_dir)