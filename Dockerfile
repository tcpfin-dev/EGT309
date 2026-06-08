FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    xgboost \
    matplotlib \
    seaborn

COPY ["ML model.py", "."]
COPY src/ src/
COPY data/ data/
COPY Saved_models/ Saved_models/

CMD python src/ingest_clean.py && python src/evaluate_savedModel.py
