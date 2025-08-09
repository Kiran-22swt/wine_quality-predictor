import numpy as np
import joblib
model=joblib.load("wine_model.pkl")
new_sample = np.array([[7.4, 0.7, 0.0, 1.9, 0.076,
                        11.0, 34.0, 0.9978, 3.51,
                        0.56, 9.4]])
predicted_quality = model.predict(new_sample)
print("Predicted wine quality: ",predicted_quality[0][0])