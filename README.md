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
python "ML model.py"
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
├── ML model.py                  # Training pipeline
├── run.sh                       # Docker build + run wrapper
├── EDA.ipynb                    # Exploratory data analysis
├── src/
│   ├── cleaning/
│   │   ├── DataCleaning.py      # Cleaning steps (depends on external `df`)
│   │   └── run_clean.py         # Self-contained entry point: loads DB, exec's DataCleaning, remaps labels
│   ├── training/
│   │   └── trainModels.py       # Empty placeholder
│   └── evaluation/
│       └── evaluate_savedModel.py  # Model evaluation script
├── data/
│   └── gas_monitoring.db        # SQLite database
└── devel/
    └── Dockerfile               # Clean + train + evaluate pipeline
```

`Saved_models/` and `gas_monitoring_cleaned.csv` are generated at runtime and gitignored.

## Model Performance

| Model | F1 Macro |
|---|---|
| Random Forest | 0.78 |
| XGBoost | 0.74 |
| Logistic Regression | 0.55 |

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

##### Step 4: Standardizing Target Formatting & Class Resampling

* **Action:** Cleaned strings within the target column to yield 3 exact unique classes: `high_activity`, `moderate_activity`, and `low_activity`. Applied SMOTE to balance classes from (1070, 5669, 3090) to (5669, 5669, 5669).

* **Logic:** Corrects parsing issues and prevents class imbalance bias during training.

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

* **Objective Function:** `multi:softmax` utilizing `mlogloss` evaluation metric.

* *Logic:* Gradient boosted trees excel at capturing non-linear boundaries.

2. **Random Forest Classifier**

* **Configuration:** Trained directly on the 80% training split with default hyperparameters.

* *Logic:* Provides an alternative using bag-based parallel tree structures.

3. **Logistic Regression**

* **Configuration:** Trained using L2 regularized coefficients with `class_weight='balanced'` across scaled arrays.

* *Logic:* Serves as a linear baseline model.

### 5. Summary of Results & Analytical Insights

Based on the evaluation benchmarks the model performance across the three architectures is as follows:

* **XGBoost & Random Forest Performance:** Tree-based ensemble architectures showed an ability to capture complex relationships across sensor inputs (F1 macro 0.74–0.78).

* **Logistic Regression Limitations:** The linear baseline model struggled to resolve overlapping classes achieving lower overall precision and recall metrics (F1 macro 0.55). This performance gap highlights that changes in indoor **Activity Level** relate to factors in a non-linear way.
