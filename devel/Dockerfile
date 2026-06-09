FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    xgboost

COPY src/ src/
COPY devel/ingest_clean.py devel/
COPY data/ data/
COPY Saved_models/ Saved_models/

CMD python devel/ingest_clean.py && python src/evaluation/evaluate_savedModel.py
