"""
Forecasting with Darts
"""
import matplotlib.pyplot as plt
import pandas as pd
from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
from darts.models import ARIMA, NaiveSeasonal, AutoARIMA
from darts.models.forecasting.nbeats import NBEATSModel
from darts.models.forecasting.rnn_model import RNNModel

series = AirPassengersDataset().load()
# series.plot()
# plt.show()

train, val = series.split_before(pd.Timestamp("19580101"))

# MinMax scaling data
transformer = Scaler()
train_transformed = transformer.fit_transform(train)
val_transformed = transformer.transform(val)
series_transformed = transformer.transform(series)

# train.plot(label="training")
# val.plot(label="validation")
# plt.show()

RANDOM_STATE = 42
N_EPOCHS = 100
BATCH_SIZE = 16
DROPOUT = 0
CHUNK_LENGTH = 10
LEARNING_RATE = 1e-3
HORIZON = 36

# Naive
naive_model = NaiveSeasonal(K=1)
naive_model.fit(train_transformed)
naive_forecast = naive_model.predict(HORIZON)

# Seasonal Naive
seasonal_model = NaiveSeasonal(K=12)
seasonal_model.fit(train_transformed)
seasonal_forecast = seasonal_model.predict(HORIZON)

# ARIMA
arima_model = ARIMA(random_state=RANDOM_STATE)
arima_model.fit(train_transformed)
arima_forecast = arima_model.predict(HORIZON)

# Improving ARIMA
trend_arima_model = ARIMA(random_state=RANDOM_STATE, trend="t")
trend_arima_model.fit(train_transformed)
trend_arima_forecast = trend_arima_model.predict(HORIZON)

# Auto ARIMA
auto_arima_model = AutoARIMA(random_state=RANDOM_STATE)
auto_arima_model.fit(train_transformed)
auto_arima_forecast = auto_arima_model.predict(HORIZON)

# LSTM
lstm_model = RNNModel(
    model="LSTM",
    hidden_dim=20,
    dropout=DROPOUT,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    optimizer_kwargs={"lr": LEARNING_RATE},
    random_state=RANDOM_STATE,
    training_length=20,
    input_chunk_length=CHUNK_LENGTH,
    force_reset=True,
    save_checkpoints=True,
)
lstm_model.fit(train_transformed)
lstm_forecast = lstm_model.predict(HORIZON)

# Transformer
nbeats_model = NBEATSModel(
    input_chunk_length=CHUNK_LENGTH,
    output_chunk_length=1,
    dropout=DROPOUT,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    random_state=RANDOM_STATE,
    optimizer_kwargs={"lr": LEARNING_RATE},
    force_reset=True,
)
nbeats_model.fit(train_transformed)
nbeats_forecast = nbeats_model.predict(HORIZON)

series_transformed.plot(label="actual")
arima_forecast.plot(label="arima forecast")
trend_arima_forecast.plot(label="trend arima forecast")
auto_arima_forecast.plot(label="auto_arima forecast")
lstm_forecast.plot(label="lstm forecast")
nbeats_forecast.plot(label="nbeats forecast")
naive_forecast.plot(label="naive forecast (K=1)")
seasonal_forecast.plot(label="naive forecast (K=12)")
plt.show()
