import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Flatten, Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



#fetch_stock_data function accepts S&P 500 tickers, the dates not dynamic by default but can easily be made so
#It then invokes a Yahoo finance function yf.download(), yfinance is imported as yf before use
#yf.download accepts start and end dates of the searched ticker data and a valid ticker currently listed
#the function if ticker is not listed throws error and does processing for invalid data stored in array
#pop(),ffill() functions to remove missing tickers and clean up non integer NAN values
#Accepts any S&P 500 ticker to query YFinance database to collect historical data
def fetch_stock_data(ticker, start="2025-01-01", end="2025-02-01"):
    ticker_data = yf.download(ticker, start=start, end=end)
    return ticker_data

#Get data for a ticker listed on S&P500
#fectch_stock_data function is called and the returned dataset is stored in a variable df
df = fetch_stock_data("MSFT")


#collected data is normalized using a scikit-learn function imported, MinMaxScaler ( ).
#The functions learn means and variance of the data set and computes the z-score
scaler = MinMaxScaler()
df_normed= scaler.fit_transform(df)

#A spatiotemporal sequence is defined
# X, y arrays are declared
# For loop is used to shape input to the convolutional ConvLSTM2D
# 1 day time step is defined for the sequences window, and target closing price is appended to y[]
#Then reshaped into a tensor input using NumPy
time_steps = 1
X, y = [], []

for i in range(len(df) - time_steps):
    X.append(df_normed[i:i+time_steps])
    y.append(df_normed[i+time_steps][0])

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1, 1, X.shape[2]))

# scikit-learn function train_test_split() is used to split all data set into
# 60% training data, 20% validation data, 20% Test data. Training data set is used to train the model,
# Validation/Development data set was used to estimate model configurations whilst development

X_train, x_split, y_train, y_split = train_test_split(X, y, test_size=0.40, random_state=1)

x_valid, X_test, y_valid, y_test = train_test_split(x_split, y_split, test_size=0.50, random_state=1)


#The ConvLSTM2D Model is imported using libraries TensorFlow and Keras
# We use Sequential ()  a Kera’s API allows to build the model layers and  sequence dense layers.
# Model has 2 fully connected dense layers, The last dense layer uses a linear activation function for prediction.

model = Sequential([
    ConvLSTM2D(filters=128, kernel_size=(1, 1), activation='relu', return_sequences=True,
               input_shape=(time_steps, 1, 1, X.shape[-1])),
    BatchNormalization(),

    ConvLSTM2D(filters=64, kernel_size=(1, 1), activation='relu', return_sequences=False),
    BatchNormalization(),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')  # Predicting stock price
])

#model was Compiled with Adam optimizer and Mean Squared Error loss function
#produced model paramters using model.summary()


print("Model summary", model.summary())

model.compile(optimizer='adam', loss='mse', metrics=["mae"])

#runs gradient descent and fits the weights to the data using x training data and y training data
history=model.fit(X_train, y_train, epochs=40, batch_size=32)

#compute mean squared error and mean absolute error
loss, mae = model.evaluate(X_test, y_test)

#use model.predict() to predict on y_test and y_train values
y_test_pred = model.predict(y_test)
y_train_pred = model.predict(y_train)

#calculates the MAE and Loss from the model.evaluate()
print(f"Test MAE : {mae:.3f}")
print(f"Test loss: {loss:.3f}")

#print statements on shape of variables used
print(f" X_training  values shape: {X_train.shape}\n")
print(f" y trainnig  values shape: {y_train.shape}\n")
print(f" y test value       shape: {y_test.shape}\n")

#
#print(tf.keras.metrics.mse(y_train, yhat).numpy())

#calculate training and test errors using the function mean_squared_error()
train_error = mean_squared_error(y_train, y_train_pred)
test_error = mean_squared_error(y_test, y_test_pred)

#print statements for the variables calculated
print(f"Training Error (MSE): {train_error:.4f}")
print(f"Test Error  (MSE)   : {test_error:.4f}")


r2 = r2_score(y_test, y_test_pred)#compute mean squared error and mean absolute error
loss, mae = model.evaluate(X_train, y_train)

y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)

print(f"Test MAE : {mae:.3f}")
print(f"Test loss: {loss:.3f}")

print(f" X_training  values shape: {X_train.shape}\n")
print(f" y trainnig  values shape: {y_train.shape}\n")
print(f" y test value       shape: {y_test.shape}\n")

#print statement for prediction
print(" y_pred:",y_test_pred)

#print(tf.keras.metrics.mse(y_train, yhat).numpy())
#R2 score values are calculated using r2_score() function from scklearn metrics library
r2 = r2_score(y_test, y_test_pred)
print(f" R2 score  (MSE): {r2:.4f}")

#calculates the RMSE value
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

#print statements for the varaibles r2 and rmse
print(f"RMSE: {rmse:.2f}")
print(f" R2 score  (MSE): {r2:.4f}")

#plots epoch  vs loss graph with x axix as Epoch and y-axis Loss
plt.plot(history.history['loss'], label='Training Loss')
#plt.plot(history.history['loss'], label='testin')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Epoch vs. Loss')
plt.show()