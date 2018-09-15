import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import datetime

from models.data_model import CurrencyDataSet
from training.train_model import LstmRNN


def main():
	raw_rate_df = pd.read_csv('currencyExchange.csv')

	with tf.Session() as sess:
		lstm_rnn = LstmRNN(
			sess
		)

		currency_data_set = CurrencyDataSet(raw_currency_df=raw_rate_df)

		currency_forecast_prediction = lstm_rnn.train(currency_data_set)

        prediction_rates = denormalize(raw_rate_df['Rate'][raw_rate_df.last_valid_index()],
                                       currency_forecast_prediction)

        prediction_dates = generate_future_dates(
            datetime.datetime.strptime(raw_rate_df['Timestamp'][raw_rate_df.last_valid_index()],
                                       "%Y-%m-%d %H:%M:%S"),
            len(prediction_rates))

        prediction_set = {"Rate": prediction_rates,
                          "Timestamp": prediction_dates}

        prediction_df = pd.DataFrame(data=prediction_set)
        raw_rate_df.append(prediction_df)

        display_chart(raw_rate_df)

        print prediction_rates, prediction_dates


def generate_future_dates(last_date, prediction_rates_length):
    return [last_date + datetime.timedelta(days=i) for i in range(prediction_rates_length)]


def denormalize(last_rate, predictions_normalized):
    denormalized_predictions = []
    for prediction in predictions_normalized:
        last_rate += prediction[0]
        denormalized_predictions.append(last_rate)

    return denormalized_predictions


def display_chart(df):
	df.plot(x='Timestamp', y='Rate', kind='line')
	plt.show()


if __name__ == '__main__':
	main()