# Air Compressor Predictive Maintenance Using LSTM

**A Time-Series Deep Learning Project for Air Compressor Failure Detection and Predictive Maintenance**

## Overview

Unexpected equipment failures are one of the major challenges in industrial environments. They can lead to unplanned downtime, increased maintenance costs, and reduced operational efficiency.

This project focuses on developing a **Predictive Maintenance system for industrial air compressors** using time-series sensor data and Deep Learning techniques, particularly Long Short-Term Memory (LSTM) networks.

The main objective is to analyze historical sensor measurements, identify abnormal operating patterns, and develop a model capable of detecting potential equipment failures based on temporal behavior.

The project uses the real-world **MetroPT-3 Air Compressor Dataset**, which contains operational data collected from an air production unit in a metro train.

## Project Objectives

- Analyze real-world industrial sensor data.
- Perform Exploratory Data Analysis (EDA) to understand normal and abnormal operating conditions.
- Identify patterns and changes in sensor behavior associated with equipment failures.
- Perform time-series preprocessing and feature engineering.
- Develop an LSTM-based model for sequential sensor data analysis.
- Evaluate model performance using appropriate classification metrics.
- Investigate the potential for early failure prediction and practical predictive maintenance applications.

## Dataset

**Dataset:** MetroPT-3 Dataset

The dataset contains approximately 1.5 million observations collected from an air production unit over a period of seven months.

| Property | Description |
|---|---|
| Dataset | MetroPT-3 |
| Equipment | Air Production Unit (Air Compressor) |
| Time period | February 1 – September 1, 2020 |
| Number of observations | 1,516,948 |
| Number of columns | 17 |
| Sampling interval | Approximately 10 seconds |
| Data type | Multivariate time-series |

### Sensor Variables

The dataset contains measurements from different sensors and operational components, including:

| Variable | Description |
|---|---|
| TP2 | Pressure at the compressor outlet |
| TP3 | Pressure in the air tank |
| H1 | Pressure-related sensor measurement |
| DV_pressure | Pressure measured in the pressure valve |
| Reservoirs | Reservoir pressure measurement |
| Oil_temperature | Compressor oil temperature |
| Motor_current | Electric motor current |
| COMP | Compressor operating status |
| DV_eletric | Electrical valve status |
| Towers | Tower operating status |
| MPG | Pressure-related operating signal |
| LPS | Low-pressure switch |
| Pressure_switch | Pressure switch status |
| Oil_level | Oil level status |
| Caudal_impulses | Airflow-related measurement |

The dataset includes both continuous sensor measurements and discrete operational signals.

## Methodology

The project follows a time-series machine learning workflow.

### 1. Data Preprocessing

The initial stage focuses on preparing the raw sensor data for analysis and modeling.

Main operations include:

- Loading and inspecting the dataset.
- Converting timestamps into datetime format.
- Checking missing values and duplicate observations.
- Investigating sensor distributions and operational behavior.
- Identifying documented equipment failure intervals.
- Resampling the original data into 10-minute intervals.

Resampling reduces the complexity of the original high-frequency data while preserving important operational patterns.

### 2. Exploratory Data Analysis (EDA)

Exploratory analysis is performed to understand how the compressor behaves under normal operating conditions and around documented failure events.

The analysis includes:

- Time-series visualization of sensor measurements.
- Statistical analysis of sensor variables.
- Investigation of correlations between operational parameters.
- Comparison of normal and abnormal operating periods.
- Analysis of sensor behavior before and during failure events.

Particular attention is given to pressure, oil temperature, and motor current because these measurements provide useful information about compressor operating conditions.

### 3. Failure Event Identification

Documented failure intervals are used to identify abnormal operating periods and construct target labels.

The dataset contains four documented failure events:

- April 18, 2020
- May 29–30, 2020
- June 5–7, 2020
- July 15, 2020

Failure intervals are incorporated into the labeling process to distinguish normal operation from failure-related observations.

The distinction between failure detection and early failure prediction is also considered during model development.

### 4. Feature Engineering

Time-series feature engineering is applied to extract useful information from sensor measurements.

The process includes statistical and temporal features such as:

- Rolling mean
- Rolling standard deviation
- Rolling minimum and maximum
- Difference between consecutive observations
- Temporal changes in sensor measurements

These features help capture changes in compressor behavior that may not be apparent from individual sensor readings.

Care is taken to preserve temporal order and avoid using future information during feature generation.

### 5. LSTM Model Development

Long Short-Term Memory (LSTM) is a type of Recurrent Neural Network (RNN) designed to learn temporal dependencies in sequential data.

Unlike conventional machine learning algorithms that generally process individual observations independently, LSTM networks can learn patterns across sequences of sensor measurements.

In this project, sequential sensor data is used as input to the LSTM model to learn temporal operating patterns and identify behavior associated with abnormal equipment conditions.

The modeling workflow includes:

- Creating sequential input samples.
- Preparing training, validation, and test datasets.
- Applying feature scaling.
- Training the LSTM network.
- Evaluating model performance on unseen time periods.

A sequence-based approach is particularly relevant for predictive maintenance because equipment degradation often develops through changes over time rather than a single abnormal measurement.

### 6. Model Evaluation

Model performance is evaluated using classification metrics suitable for imbalanced industrial failure datasets.

The evaluation focuses on:

| Metric | Purpose |
|---|---|
| Precision | Measures how many predicted failures are actually failures |
| Recall | Measures how many actual failures are detected |
| F1-Score | Balances precision and recall |
| ROC-AUC | Measures the model's ability to distinguish between classes |
| PR-AUC | Evaluates classification performance under class imbalance |

In addition to conventional classification metrics, the project considers practical maintenance-related measurements:

- Number of detected failures.
- Missed failure events.
- False alarms.
- Detection timing relative to failure events.

For predictive maintenance, detecting a failure is not sufficient. The system should ideally identify abnormal behavior early enough to allow maintenance teams to take preventive action.

## Technologies and Libraries

The project is developed using Python and the following technologies:

| Technology | Application |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Scikit-learn | Data preprocessing and evaluation |
| TensorFlow / Keras | LSTM model development |
| Jupyter Notebook | Exploratory analysis and experimentation |

## Project Workflow

The overall workflow can be summarized as follows:

```text
Raw Sensor Data
       |
       v
Data Loading and Inspection
       |
       v
Data Preprocessing
       |
       v
Exploratory Data Analysis
       |
       v
Failure Event Identification
       |
       v
Feature Engineering
       |
       v
Time-Series Sequence Generation
       |
       v
LSTM Model Development
       |
       v
Model Training and Validation
       |
       v
Performance Evaluation
       |
       v
Failure Detection and Predictive Maintenance
```

## Key Challenges

Several challenges are considered throughout this project:

- **Class Imbalance:** Failure-related observations represent a small proportion of the entire dataset.
- **Temporal Dependencies:** Sensor measurements are time-dependent and must be processed while preserving chronological order.
- **Data Leakage:** Training and evaluation procedures must prevent information from future observations from influencing model training.
- **Limited Failure Events:** The dataset contains only four documented failure events, which limits the statistical reliability of generalization and early prediction estimates.
- **False Alarms:** Excessive false alarms can reduce the practical value of a predictive maintenance system.

## Future Improvements

Potential future developments include:

- Comparing LSTM performance with conventional machine learning models such as XGBoost.
- Improving early failure prediction using different prediction horizons.
- Developing more robust anomaly detection methods.
- Investigating model interpretability to understand which sensor variables contribute most to failure detection.
- Integrating the trained model into a FastAPI-based application.
- Developing a monitoring dashboard for visualizing sensor measurements and model predictions.
- Exploring real-time sensor data processing for industrial applications.

## Project Motivation

This project is part of my ongoing learning and research in **Machine Learning, Deep Learning, Time-Series Analysis, and Industrial Predictive Maintenance**.

My broader objective is to explore how AI-based solutions can improve industrial equipment reliability by identifying abnormal operating behavior and supporting maintenance decisions.

The project provides a practical opportunity to work with real-world industrial data and develop an end-to-end machine learning workflow for equipment condition monitoring.

## Author

**Mostafa Mousivand**

- GitHub: [MostafaMousivand](https://github.com/MostafaMousivand)

---

*This is an ongoing portfolio project. The modeling pipeline and predictive maintenance capabilities are being developed and evaluated progressively.*
