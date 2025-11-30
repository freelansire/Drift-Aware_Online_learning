import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_online_lstm(input_dim):
    model = Sequential([
        LSTM(16, return_sequences=False, input_shape=(20, input_dim)),
        Dense(8, activation='relu'),
        Dense(input_dim)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def make_window(series, window_size=20):
    if len(series) < window_size:
        return None
    arr = np.array(series[-window_size:])
    return arr.reshape(1, window_size, arr.shape[1])
