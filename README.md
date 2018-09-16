# traveller-currency-prediction
An application that forecasts future currency exchange rates based on previous exchange rate patterns using a recurrent neural netwo preventork that makes use of LSTM cells. (Credit to @LilianWeng for tutorial on stock prediction, training logic is based off of that)

## Usage
Execute `training.py` to scrape data from xe.com. After the data has been loaded into the csv file, rename it to `currencyExchange.csv` and execute `main.py`, which will train the model.
To prevent the model from overfitting the data, `training.py` needs to be modified to use multiple currency rate pairs. And training needs to be done seperately for each of them. This will allow the network to make more general predictions as to how currency changes, rather than specific to the given currency pair.
