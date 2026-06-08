# EGT309 — Gas Sensor Activity Classification

Indoor activity level classification from gas sensor and environmental readings (10,000-row SQLite dataset).

## Setup

```bash
pip install pandas scikit-learn xgboost matplotlib seaborn
```

## Run

### Ingest & clean data

```bash
python src/ingest_clean.py
```

Reads from `data/gas_monitoring.db`, cleans and exports `gas_monitoring_cleaned.csv`.

### Train models

```bash
python "ML model.py"
```

### Evaluate models

```bash
python src/evaluate_savedModel.py
```

Loads saved models from `Saved_models/` and prints F1 macro scores plus classification reports.

### EDA

```bash
jupyter notebook EDA.ipynb
```

### Docker

```bash
docker build -t egt309 .
docker run egt309
```

## Data

SQLite database (`data/gas_monitoring.db`) with the following sensor columns:

| Feature | Type |
|---|---|
| Temperature, Humidity | Environmental |
| CO2 (Infrared + ElectroChemical) | Gas sensors |
| Metal Oxide sensors (4 units) | Gas sensors |
| CO sensor | Gas sensors |
| Time of Day, HVAC Mode, Ambient Light | Context |

**Target**: `Activity Level` (Low / Moderate / High)

## Project Structure

```
EGT309/
  ML model.py                  # Training pipeline
  EDA.ipynb                    # Exploratory data analysis
  src/
    ingest_clean.py             # Data ingestion & cleaning
    evaluate_savedModel.py      # Model evaluation script
  Saved_models/                 # Trained model artifacts
  data/
    gas_monitoring.db           # SQLite database
```

## License

MIT
