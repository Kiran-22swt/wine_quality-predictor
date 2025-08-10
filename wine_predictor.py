import numpy as np
import joblib
model=joblib.load("wine_model.pkl")
sample = np.array([[7.4, 0.7, 0.0, 1.9, 0.076,
                        11.0, 34.0, 0.9978, 3.51,
                        0.56, 9.4]])
result= model.predict(sample)
print("result: ",result[0][0])