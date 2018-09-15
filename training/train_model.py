import tensorflow as tf


class LstmRNN(object):
	def __init__(self, sess,
				 lstm_size=128,
				 num_layers=1,
				 num_steps=30,
				 input_size=1):

		self.sess = sess
		self.lstm_size = lstm_size
		self.num_layers = num_layers
		self.num_steps = num_steps
		self.input_size = input_size

		self.build_graph()

	def build_graph(self):

		self.learning_rate = tf.placeholder(tf.float32, None,
												name="learning_rate")
		self.keep_prob = tf.placeholder(tf.float32, None, name="keep_prob")

		self.inputs = tf.placeholder(tf.float32,
									 [None, self.num_steps, self.input_size],
									 name="inputs")
		self.targets = tf.placeholder(tf.float32, [None, self.input_size],
									  name="targets")

		def _create_lstm_cell():
			lstm_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_size)
			lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
				cell=lstm_cell,
				output_keep_prob=self.keep_prob
			)
			return lstm_cell

		stacked_lstm = tf.contrib.rnn.MultiRNNCell(
			[_create_lstm_cell() for _ in range(self.num_layers)],
			state_is_tuple=True
		) if self.num_layers > 1 else _create_lstm_cell()

		# Convert to dynamic RNN to have variable sequence length
		val, state_ = tf.nn.dynamic_rnn(stacked_lstm,
										inputs=self.inputs,
										dtype=tf.float32, scope="dynamic_rnn")

		# Before transpose, val.get_shape() = (batch_size, num_steps, lstm_size)
		# After transpose, val.get_shape() = (num_steps, batch_size, lstm_size)
		val = tf.transpose(val, [1, 0, 2])

		last = tf.gather(val, int(val.get_shape()[0]) - 1, name="lstm_state")
		weights = tf.Variable(tf.truncated_normal([self.lstm_size,
												   self.input_size]),
							  name="weights")

		biases = tf.Variable(tf.constant(0.1,
									   shape=[self.input_size]),
						   			   name="biases")

		self.pred = tf.matmul(last, weights) + biases

		self.loss = tf.reduce_mean(tf.square(self.pred - self.targets),
								   name="loss_mse_train")

		self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).\
			minimize(self.loss, name="rmsprop_optimizer")

		# Returns all trainable parameters
		self.t_vars = tf.trainable_variables()

		# Creates a saver to save the model
		self.saver = tf.train.Saver()

	def train(self, currency_data_set):
		tf.global_variables_initializer().run()

		total_steps = 0
		# Diminish the learning rate over time after the 5th epoch
		# to avoid over stepping
		learning_rates_to_use = [
			0.001* (
					0.99 ** max(
				float(i + 1 - 5), 0.0)
			) for i in range(100)]

		for epoch in range(100):
			curr_lr = learning_rates_to_use[epoch]

			for batch_x, batch_y in currency_data_set.generate_one_epoch(64):

				train_feed = {
					self.inputs: batch_x,
					self.keep_prob: 0.8,
					self.targets: batch_y,
					self.learning_rate: curr_lr
				}

				train_loss = self.sess.run([self.loss, self.optimizer], train_feed)
				total_steps += 1
				print "Training loss: {}\n Epoch Number: {}".format(train_loss,
																	total_steps)

		self.saver.save(self.sess, "trained_models/currency-prediction",
						global_step=100)

		test_batch_x, test_batch_y = currency_data_set.generate_test_data(64)
		test_data_feed = {
			self.learning_rate: 0.0,
			self.keep_prob: 1.0,
			self.inputs: test_batch_x,
			self.targets: test_batch_y
		}

		final_pred, final_loss = self.sess.run([self.pred, self.loss],
											   test_data_feed)

		print final_pred, final_loss, total_steps
		return final_pred
