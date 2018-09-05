import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf


def main():
	display_chart()
	# lstm_rnn = LstmRNN(
	#
	# )


class LstmRNN(object):
	def __init__(self, sess, stock_count,
				 lstm_size=128,
				 num_layers=1,
				 num_steps=30,
				 input_size=1,
				 embed_size=None):
		self.sess = sess
		self.stock_count = stock_count

		self.lstm_size = lstm_size
		self.num_layers = num_layers
		self.num_steps = num_steps
		self.input_size = input_size

		self.use_embed = (embed_size is not None) and (embed_size > 0)
		self.embed_size = embed_size or -1

		self.build_graph()

	def build_graph(self):

		def _create_lstm_cell():
			lstm_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_size)
			lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
				cell=lstm_cell,
				output_keep_prob=self.keep_prob
			)
			return lstm_cell

		cell = tf.contrib.rnn.MultiRNNCell(
			[_create_lstm_cell() for _ in range(self.num_layers)],
			state_is_tuple=True
		) if self.num_layers > 1 else _create_lstm_cell()

		self.learning_rate = tf.placeholder(tf.float32, None,
												name="learning_rate")
		self.keep_prob = tf.placeholder(tf.float32, None, name="keep_prob")

		self.inputs = tf.placeholder(tf.float32,
									 [None, self.num_steps, self.input_size],
									 name="inputs")
		self.targets = tf.placeholder(tf.float32, [None, self.input_size],
									  name="targets")

		# Run dynamic RNN
		val, state_ = tf.nn.dynamic_rnn(cell,
										inputs=self.inputs,
										dtype=tf.float32, scope="dynamic_rnn")

		# Before transpose, val.get_shape() = (batch_size, num_steps, lstm_size)
		# After transpose, val.get_shape() = (num_steps, batch_size, lstm_size)
		print val.get_shape()
		val = tf.transpose(val, [1, 0, 2])
		print val.get_shape()

		last = tf.gather(val, int(val.get_shape()[0]) - 1, name="lstm_state")
		


def display_chart():
	df = pd.read_csv('currencyExchange.csv')
	df.plot(x='Timestamp',y='Rate', kind="line")

	plt.show()


if __name__ == '__main__':
	main()
