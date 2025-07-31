import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def main():
    model = joblib.load("model.joblib")
    X, y = fetch_california_housing(return_X_y=True)
    _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    preds = model.predict(X_test[:5])
    print("Sample Predictions:")
    for i, pred in enumerate(preds, start=1):
        print(f"Prediction {i}: {pred:.4f}")

if __name__ == "__main__":
    main()

