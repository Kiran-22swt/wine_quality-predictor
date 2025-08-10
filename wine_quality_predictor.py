import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tflow
from tensorflow import keras
import joblib
mydata = pd.read_csv('winequality-red.csv') 
x = mydata[["fixed acidity", "volatile acidity", "citric acid","residual sugar", "chlorides", "free sulfur dioxide","total sulfur dioxide", "density", "pH","sulphates", "alcohol"]]
y = mydata["quality"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1) 
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
y_pred = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mean = mean_absolute_error(y_test, y_pred)
print("RMSE: ",rmse)
print("Mean Absolute Error: ",mean)
joblib.dump(model,"wine_model.pkl")