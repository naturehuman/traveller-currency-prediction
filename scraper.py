import time
import datetime
import csv

from decimal import Decimal

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains


def fill_missing_timesteps(incomplete_data):
    complete_data = []
    i = 0
    j = 1
    while j < len(incomplete_data):
        day_difference = (incomplete_data[j][1] - incomplete_data[i][1]).days
        if day_difference == 1:
            complete_data.append([round(incomplete_data[i][0], 5),
                                  incomplete_data[i][1]])
        else:
            rate_increment = \
                (Decimal(incomplete_data[j][0]) -
                 Decimal(incomplete_data[i][0]))/day_difference
            timestamp = incomplete_data[i][1]
            for k in range(day_difference):
                complete_data.append([round(incomplete_data[i][0] +
                                      k*rate_increment, 5),
                                      timestamp])
                timestamp += datetime.timedelta(days=1)
        i += 1
        j += 1
    complete_data.append(incomplete_data[-1])

    return complete_data


driver = webdriver.Firefox()
driver.get("https://www.xe.com/currencycharts/?from=GBP&to=CAD&view=2Y")
time.sleep(5)

csvfile = "currencyExchanges.csv"

try:
  p_element = driver.find_element_by_id("rates_detail_desc")
  timestamp = driver.find_element_by_id("rates_detail_title")
 
except AttributeError: 
  print "Could not find Element"

action = ActionChains(driver)
rates_detail_chart = driver.find_element_by_id("rates_detail_event")

chart_width = int(rates_detail_chart.size['width'])

data = []

for mouse_x in range(chart_width):
    action.move_to_element(rates_detail_chart)
    action.move_by_offset(-chart_width/2, 0)
    action.move_by_offset(mouse_x, 0).perform()
    currency_rate = p_element.text[14:21]
    data.append([Decimal(currency_rate), (datetime.datetime.strptime(
        timestamp.text[:16],'%d %b %Y %H:%M'))])
    action = ActionChains(driver)

# Filter all data that is greater than last date
data = fill_missing_timesteps(filter(lambda item: item[1] > datetime.datetime(2016, 8, 19, 0, 0), data))

# Assuming data is a list of lists
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(["Rate", "Timestamp"])
    writer.writerows(data)

driver.close()
