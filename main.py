# hfskdjfhsdkjf
from darts.utils.missing_values import fill_missing_values
from darts import TimeSeries
from darts.datasets import AirPassengersDataset
from darts.models import ARIMA

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

series = AirPassengersDataset().load()
series.plot()
plt.show()

values = np.arange(50, step=0.5)
values[10:30] = np.nan
values[60:95] = np.nan
series_ = TimeSeries.from_values(values)

(series_ - 10).plot(label="with missing values (shifted below)")
fill_missing_values(series_).plot(label="without missing values")

train, val = series.split_before(pd.Timestamp("19580101"))
train.plot(label="training")
val.plot(label="validation")
plt.show()

naive_model = ARIMA()
naive_model.fit(train)
naive_forecast = naive_model.predict(36)

series.plot(label="actual")
naive_forecast.plot(label="naive forecast (K=1)")
plt.show()
