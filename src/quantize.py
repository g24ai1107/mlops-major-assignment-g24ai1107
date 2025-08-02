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

# Evaluate original model
y_pred_orig = model.predict(X_test)
r2_orig = r2_score(y_test, y_pred_orig)
mse_orig = mean_squared_error(y_test, y_pred_orig)

print("Original Model Performance")
print(f"R² Score: {r2_orig:.4f}")
print(f"MSE: {mse_orig:.4f}")

# Extract coefficients and intercept
coef = model.coef_
intercept = model.intercept_

# Save unquantized params
joblib.dump((coef, intercept), "unquant_params.joblib")

# Initialize quantized arrays
q_coef = []
scales = []
zero_point = 128  
for c in coef:
    scale = max(abs(c), 1e-8) / 127  # Avoid div by zero
    q = np.clip(np.round(c / scale) + zero_point, 0, 255).astype(np.uint8)
    q_coef.append(q)
    scales.append(scale)

# Intercept
scale_i = max(abs(intercept), 1e-8) / 127
q_intercept = np.clip(np.round(intercept / scale_i) + zero_point, 0, 255).astype(np.uint8)

# Save quantized parameters
joblib.dump((q_coef, q_intercept, scales, scale_i, zero_point), "quant_params.joblib")

# Dequantize
dq_coef = [(q.astype(np.float32) - zero_point) * scale for q, scale in zip(q_coef, scales)]
dq_intercept = (q_intercept.astype(np.float32) - zero_point) * scale_i

# Predict using quantized weights
y_pred_quant = np.dot(X_test, dq_coef) + dq_intercept
r2_quant = r2_score(y_test, y_pred_quant)
mse_quant = mean_squared_error(y_test, y_pred_quant)

print("\nQuantized Model Performance")
print(f"R² Score: {r2_quant:.4f}")
print(f"MSE: {mse_quant:.4f}")

# File size comparison
print("\nModel Size Comparison")
print(f"Original model.joblib: {os.path.getsize('model.joblib') / 1024:.2f} KB")
print(f"Quantized quant_params.joblib: {os.path.getsize('quant_params.joblib') / 1024:.2f} KB")

