FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run train.py first to generate model.joblib
RUN python src/train.py

# Run prediction using the trained model
CMD ["python", "src/predict.py"]

