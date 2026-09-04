# Predictive Maintenance & Anomaly Detection
## 6G-Integrated Smart Manufacturing — Thales Group

## Project Overview
Anomaly detection and predictive maintenance analytics for a 6G-integrated 
smart manufacturing system covering 50 industrial machines from January to 
March 2025. The system detects abnormal sensor behavior using Isolation Forest 
and classifies machine health into Low, Medium, and High risk levels.

## Project Structure

    Thales_Manufacturing_Analytics/
    ├── data/
    │   ├── Thales_Group_Manufacturing.csv
    │   ├── processed_manufacturing_data.csv
    │   ├── active_anomaly_data.csv
    │   └── high_risk_machines.csv
    ├── notebooks/
    │   └── EDA_and_Analysis.ipynb
    ├── outputs/
    │   └── charts/
    ├── Reports/
    │   ├── research_paper.pdf
    │   └── executive_summary.pdf
    ├── app.py
    ├── requirements.txt
    └── README.md

## Setup & Installation

    pip install -r requirements.txt

## Running the Dashboard

    streamlit run app.py

## Key Findings

1. 3,503 anomalies detected across 70,054 active machine records (5% contamination rate).
2. Machine 8 recorded the highest peak anomaly score of 0.9432.
3. Machine 3 had the most high-risk events (25) across the observation window.
4. A notable risk spike was detected on March 1, 2025 across multiple machines.
5. High-risk events occur uniformly across all hours — no single shift is disproportionately risky.
6. Production Speed and Error Rate are the strongest discriminators of machine efficiency status.

## Methodology

- Baseline behavior modeled per Machine_ID using 60-minute rolling windows
- 17 engineered features: sensor deviations, vibration-power ratio, error escalation trend
- Isolation Forest: 200 estimators, 5% contamination, StandardScaler preprocessing
- Risk classification: High (>=0.7), Medium (>=0.4), Low (<0.4)

## KPIs Tracked

| KPI | Description |
|-----|-------------|
| Anomaly Score | Degree of deviation from normal behavior (0-1) |
| Maintenance Risk Level | Low / Medium / High classification |
| High-Risk Machine List | Priority inspection targets |
| Early Warning Lead Time | Time before potential failure |
| Downtime Prevention Index | Estimated risk hours per machine |

## Data Source
Thales Group — 6G-Integrated Smart Manufacturing Dataset (Jan–Mar 2025)

## Tools Used
- Python, Pandas, NumPy — data processing
- Scikit-learn — Isolation Forest anomaly detection
- Matplotlib, Seaborn — static visualizations
- Plotly, Streamlit — interactive dashboard