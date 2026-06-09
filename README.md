# EGT309 — Gas Sensor Activity Classification

Indoor activity level classification from gas sensor and environmental readings (10,000-row SQLite dataset).

## Setup

```bash
pip install pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn
```

## Run

### Ingest & clean data

```bash
python src/cleaning/run_clean.py
```

Reads from `data/gas_monitoring.db`, cleans and exports `gas_monitoring_cleaned.csv`.

### Train models

```bash
python src/training/trainModels.py
```

Trains XGBoost, Random Forest, and Logistic Regression. Saves artifacts to `Saved_models/`.

### Evaluate models

```bash
python src/evaluation/evaluate_savedModel.py
```

Loads saved models from `Saved_models/` and prints F1 macro scores plus classification reports.

### Full pipeline (Docker)

```bash
./run.sh
```

Or manually:

```bash
docker build -t egt309 -f devel/Dockerfile .
docker run --rm egt309
```

Runs clean → train → evaluate end-to-end. Models are generated at runtime (not stored in the repo).

### EDA

```bash
jupyter notebook EDA.ipynb
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
├── run.sh                       # Docker build + run wrapper
├── EDA.ipynb                    # Exploratory data analysis
├── src/
│   ├── cleaning/
│   │   ├── DataCleaning.py      # Cleaning steps (depends on external `df`)
│   │   └── run_clean.py         # Self-contained entry point: loads DB, exec's DataCleaning, remaps labels
│   ├── training/
│   │   └── trainModels.py       # Training pipeline (SMOTE + hyperparameter tuning + early stopping)
│   └── evaluation/
│       └── evaluate_savedModel.py  # Model evaluation script
├── data/
│   └── gas_monitoring.db        # SQLite database
└── devel/
    └── Dockerfile               # Clean + train + evaluate pipeline
```

`Saved_models/` and `gas_monitoring_cleaned.csv` are generated at runtime and gitignored.

## Model Performance

Evaluation on 1,966 test samples (20% holdout) after SMOTE oversampling:

| Model | F1 Macro | Accuracy | Notes |
|---|---|---|---|
| XGBoost | 0.89 | 0.91 | Early stopping (50 rounds), `eta=0.05`, `max_depth=7` |
| Random Forest | 0.82 | 0.84 | 1,000 estimators, `max_depth=50`, `class_weight='balanced'` |
| Logistic Regression | 0.50 | 0.58 | Linear baseline, `class_weight='balanced'`, `max_iter=1000` |

## License

MIT


## Comprehensive Project Documentation: EDA and Cleaning

### 1. Initial Data Exploration (EDA)

The goal is to assess the baseline structure, integrity, distributions and flaws in the initial `gas_monitoring` SQL database.

#### Initial State Analysis

* Volume: 10000 rows, 14 data columns.

* Feature Schema: 9 numeric metrics, 5 categorical variables.

* Immediate Anomalies Uncovered:

     1. Impossible Outliers: Temperature readings peaked at an impossible 307.07°C while relative humidity hit negative numbers and extreme maximums.

     2. Sensor Faults: `CO2_InfraredSensor` returned impossible negative concentration values.

     3. High Cardinality and Syntactic Noise: `HVAC Operation Mode` had formatting fragmentation.

     4. Target Irregularity: `Activity Level` was syntactically inconsistent.

### 2. Advanced Data Cleaning, Standardisation & Imputation

The goal is to clean the sensor artifacts, drop redundant information, resolve missing values, and isolate true tracking features.

#### Process Step-by-Step & Underlying Logic

##### Step 1: Handling Environmental Sensor Extremes

* **Action:** 795 temperature records, 414 humidity records and 113 infrared CO2 values set to `NaN`.

* **Logic:** These values were replaced using **session-level medians** to maintain local coherence.

##### Step 2: Categorical Feature Repair & Structural Mode Imputation

* **Action:** Resolved missing entries for `Ambient Light Level` using its mode value. Standardized `HVAC Operation Mode` from text variants to 6 core operations.

* **Logic:** Consolidating text variants aligns states into distinct categorical groups.

##### Step 3: Removing Redundant Features & Sensor De-duplication

* **Action:** Dropped `MetalOxideSensor_Unit2` and `CO2_ElectroChemicalSensor` from features.

* **Logic:** Unit 2 had 14% missing values and was highly collinear with `MetalOxideSensor_Unit4`. The infrared CO2 sensor outperforms the electrochemical variant.

##### Step 4: Standardizing Target Formatting

* **Action:** Cleaned strings within the target column to yield 3 exact unique classes: `high_activity`, `moderate_activity`, and `low_activity`.

* **Logic:** Corrects parsing issues and ensures consistent categorical labels.

**Note:** SMOTE oversampling was removed from the cleaning pipeline and relocated to `src/training/trainModels.py`, where it is applied after feature scaling and before the train/test split.

### 3. End-to-End Machine Learning Pipeline

The goal is to ingest the data, process features mathematically, partition data properly, and train predictive classifiers.

#### Feature Transformation Architecture

1. **Target Conversion:** Used a `LabelEncoder` to convert the string categories of the target column into a 3-class index map.

2. **Categorical Feature Encoding:** Processed categorical features via `pd.get_dummies(..., drop_first=True)`.

* *Logic:* Dropping the first dummy column prevents multi-collinearity.

3. **Feature Scaling:** Applied a `StandardScaler` to normalize the encoded training dataset.

* *Logic:* Ensures that features with different scales do not dominate distance measurements or gradient calculations.

#### Data Partitioning Strategy

* **Partition Splits:** The data was split into an 80% training set and a 20% test set using stratified sampling.

### 4. Model Training & Evaluation Setup

The goal is to evaluate performance across model architectures.

#### Candidate Model Configurations

1. **XGBoost Classifier**

* **Configuration:** Trained with `xgb.train()` using DMatrix format, early stopping (50 rounds), learning rate 0.05, max depth 7, and up to 1,000 boosting rounds on SMOTE-balanced data.

* *Logic:* Gradient boosted trees excel at capturing non-linear boundaries. Early stopping prevents overfitting.

2. **Random Forest Classifier**

* **Configuration:** Trained with 1,000 estimators, max depth 50, min samples per leaf 4, and `class_weight='balanced'` on SMOTE-balanced data.

* *Logic:* Provides an alternative using bag-based parallel tree structures. Tuned depth and leaf constraints reduce overfitting.

3. **Logistic Regression**

* **Configuration:** Trained using L2 regularized coefficients with `class_weight='balanced'` across scaled arrays.

* *Logic:* Serves as a linear baseline model. Performance gap vs. tree ensembles confirms non-linear class boundaries.

### 5. Summary of Results & Analytical Insights

Based on the evaluation benchmarks the model performance across the three architectures is as follows:

* **XGBoost (Best):** Achieved 0.89 F1 macro and 0.91 accuracy with early stopping and hyperparameter tuning. Strong across all three classes, with "Low Activity" precision at 0.95.

* **Random Forest:** Achieved 0.82 F1 macro and 0.84 accuracy. Performs well on "Moderate Activity" (recall 0.88) and "High Activity" (recall 0.86) but lags XGBoost on "Low Activity".

* **Logistic Regression (Baseline):** Achieved 0.50 F1 macro and 0.58 accuracy. Struggles severely with "High Activity" (F1 0.28), confirming that indoor activity levels relate to sensor inputs in a highly non-linear way.
