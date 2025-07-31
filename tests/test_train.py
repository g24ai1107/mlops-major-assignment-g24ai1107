from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Setup
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def test_model_training():
    model = LinearRegression()
    model.fit(X_train, y_train)
    assert isinstance(model, LinearRegression)

def test_model_has_coefficients():
    model = LinearRegression()
    model.fit(X_train, y_train)
    assert hasattr(model, "coef_") and model.coef_ is not None

def test_model_r2_score_above_threshold():
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    assert r2 > 0.5, f"Expected R² > 0.5, got {r2}"

