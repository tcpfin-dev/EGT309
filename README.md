# EGT309 — Gas Sensor Activity Classification

Indoor activity level classification from gas sensor and environmental readings (10,000-row SQLite dataset).

## Setup

```bash
pip install pandas scikit-learn xgboost matplotlib seaborn
```

## Run

```bash
python main.py         # loads DB, prints summary stats
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

## ML Output

Pre-generated plots in `plots/` show classification results using Logistic Regression, Random Forest, and XGBoost.

## License

MIT
