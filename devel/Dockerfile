FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    xgboost \
    imbalanced-learn

COPY src/ src/

COPY data/ data/
CMD python src/cleaning/run_clean.py && python src/training/trainModels.py && python src/evaluation/evaluate_savedModel.py
