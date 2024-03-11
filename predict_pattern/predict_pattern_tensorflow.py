import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# load the csv file content into dataframe
df = pd.read_csv("pattern.csv")
arrays = [np.array(df[col]) for col in df.columns]

for arr in arrays:
    sequence = arr

    # Prepare the dataset
    # all elements except the last one
    X = sequence[:-1]
    # all elements except the first one
    y = sequence[1:]

    # Reshape the data for the model
    X = X.reshape((len(X), 1))
    y = y.reshape((len(y), 1))

    # Normalize the data
    X = X / float(len(sequence))
    y = y / float(len(sequence))

    # Define the model
    model = Sequential([
        Dense(50, activation='relu', input_dim=1),
        Dense(1)
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

    # Fit the model
    model.fit(X, y, epochs=1000, verbose=0)

    # Predict the next number
    last_num = sequence[-1] / float(len(sequence))  # Normalize the last number
    next_num = model.predict(np.array([[last_num]])) * len(sequence)  # Predict and denormalize

    print(f"The next number in the sequence of {arr} is: {next_num[0][0]:.0f}")
