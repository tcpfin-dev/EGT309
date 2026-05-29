FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    xgboost \
    matplotlib \
    seaborn

COPY main.py .
COPY data/ data/

CMD ["python", "main.py"]
