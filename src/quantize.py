import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import os

# Load dataset
X, y = fetch_california_housing(return_X_y=True)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load trained model
model = joblib.load("model.joblib")

# Inference before quantization
y_pred_orig = model.predict(X_test)
r2_orig = r2_score(y_test, y_pred_orig)
mse_orig = mean_squared_error(y_test, y_pred_orig)

print("Original Model Performance")
print(f"R² Score: {r2_orig:.4f}")
print(f"MSE: {mse_orig:.4f}")

# Save raw parameters
coef = model.coef_
intercept = model.intercept_
joblib.dump((coef, intercept), "unquant_params.joblib")

# Quantize using int8
max_val = np.max(np.abs(coef))
scale = 127.0 / max_val
q_coef = np.round(coef * scale).astype(np.int8)
q_intercept = int(round(intercept * scale))

# Save quantized
joblib.dump((q_coef, q_intercept, scale), "quant_params.joblib")

# Dequantize
dq_coef = q_coef.astype(np.float32) / scale
dq_intercept = q_intercept / scale

# Inference after quantization
y_pred_quant = np.dot(X_test, dq_coef) + dq_intercept
r2_quant = r2_score(y_test, y_pred_quant)
mse_quant = mean_squared_error(y_test, y_pred_quant)

print("\nQuantized Model Performance")
print(f"R² Score: {r2_quant:.4f}")
print(f"MSE: {mse_quant:.4f}")

# Compare file sizes
print("\nModel Size Comparison")
print(f"Original: {os.path.getsize('model.joblib') / 1024:.2f} KB")
print(f"Quantized: {os.path.getsize('quant_params.joblib') / 1024:.2f} KB")

