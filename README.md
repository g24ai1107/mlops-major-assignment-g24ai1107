#  MLOps Major Assignment - Linear Regression

This project demonstrates a complete MLOps pipeline for training, testing, quantizing, Dockerizing, and deploying a Linear Regression model using the California Housing dataset. It also integrates CI/CD using GitHub Actions.

---

##  Project Structure

mlops-linear-regression/
├── src/
│   ├── train.py
│   ├── predict.py
│   └── quantize.py
├── tests/
│   └── test_train.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── requirements.txt
├── .gitignore
├── environment.yml
└── README.md

---

##  Step-by-Step Workflow

###  Step 1: Initial Setup
- Created and activated a conda environment `mlops_env`.
- Initialized a GitHub repository.
- Added basic project files (`README.md`, `.gitignore`, `requirements.txt`).

###  Step 2: Model Training
- Used the California Housing dataset from `sklearn.datasets`.
- Trained a `LinearRegression` model using `scikit-learn`.
- Calculated and displayed R² and MSE metrics.
- Saved the model as `model.joblib`.

###  Step 3: Prediction
- Loaded `model.joblib` and ran predictions on sample test data.
- Displayed a few sample outputs to verify correctness.

###  Step 4: Quantization
- Quantizes model weights and intercept using `uint8`.
- Uses per-parameter scale factors for better precision.
- Saves quantized params in `quant_params.joblib`.
- Evaluates R² and MSE before and after quantization.

###  Step 5: Dockerization
- Wrote a Dockerfile that installs dependencies, trains the model, and performs predictions.
- Ensures reproducibility of model training and testing in containerized environments.

###  Step 6: GitHub Actions CI/CD
- Configured GitHub Actions to automate:
  - Running unit tests using `pytest`
  - Training the model and performing quantization
  - Building and running the Docker container

---

##  Model Performance & Size Comparison

| Metric             | Original Model    | Quantized Model    |
|--------------------|-------------------|--------------------|
|  R² Score          | 0.5758            | 0.5758             |
|  MSE               | 0.5559            | 0.5559             |
|  Model Size        | 0.68 KB           | 0.40 KB            |

> The quantized model reduces size by over 50% while maintaining comparable accuracy, demonstrating efficient trade-offs in model deployment.

---

##  Tech Stack

- **Language**: Python 3.10
- **Libraries**: scikit-learn, joblib, pytest
- **Packaging**: Conda, Docker
- **CI/CD**: GitHub Actions

---

Chandan Jena  
G24AI1107
