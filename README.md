# Rossmann Store Sales Forecasting

![Python](https://img.shields.io/badge/Python-3.14-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green)
![Dash](https://img.shields.io/badge/Plotly_Dash-4.1.0-orange)
![R2 Score](https://img.shields.io/badge/R²_Score-0.9252-brightgreen)

An end-to-end Data Science project combining **Sales Analytics** 
and **Machine Learning** to forecast Rossmann store sales across 
1,115 stores in Germany.

---

## Live Dashboard
🔗 [View Live Dashboard](#) ← update after deployment

---

## Project Overview

| Component | Description |
|---|---|
| **Dataset** | Rossmann Store Sales — Kaggle |
| **Rows** | 1,017,209 sales records |
| **Stores** | 1,115 stores across Germany |
| **Period** | Jan 2013 — Jul 2015 |
| **Model** | LightGBM (Gradient Boosting) |
| **R² Score** | 0.9252 (92.5% accuracy) |
| **MAPE** | 8.97% avg error |

---

## Project Structure

```
rossmann-sales-forecasting/
├── notebooks/
│   ├── 01_EDA.ipynb                 # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb # Feature Engineering
│   ├── 03_Model_Training.ipynb      # ML Model Training
│   └── 04_Dashboard.ipynb           # Dashboard Exploration
├── dashboard/
│   ├── app.py                       # Plotly Dash App
│   ├── Procfile                     # Render deployment
│   └── requirements.txt             # Dependencies
├── src/
│   └── model.pkl                    # Trained LightGBM model
├── reports/
│   ├── 01_actual_vs_predicted.png
│   ├── 02_feature_importance.png
│   ├── 03_forecast_vs_actual.png
│   └── 04_future_forecast.png
└── data/
├── raw/                         # Original Kaggle data
└── processed/                   # Cleaned & engineered data
```

---

## Key Business Insights

- December is peak sales month — Christmas drives **€231M** revenue
- Promotions drive **+38.8% sales uplift** and **+21.2% more customers**
- Store Type A is most promo-sensitive — **+43% uplift** during promos
- Easter & Christmas holidays drive **+40% sales** above normal days
- **8x gap** between best (Store 262, €19.5M) and worst store (€2.4M)
- Sunday + November/December = highest sales combination (**€11,788/day**)

---

## ML Model Performance

| Metric | Baseline | LightGBM | Improvement |
|---|---|---|---|
| RMSE | €3,118 | €851 | 72.7% |
| MAE | €2,280 | €605 | 73.5% |
| MAPE | 37.09% | 8.97% | 75.8% |
| R² | -0.005 | 0.925 | — |

---

## Dashboard Features

- **KPI Overview** — Total sales, avg daily sales, customers, best store
- **Monthly Trend** — Year over year sales comparison
- **Day of Week Analysis** — Best performing days
- **Store Performance** — Top 15 stores ranking
- **Promo Impact** — Promotion vs non-promotion analysis
- **Actual vs Predicted** — Model test results visualization
- **Future Forecast** — ML powered predictions (4-16 weeks ahead)

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.14 | Core language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Static visualizations |
| Plotly Dash | Interactive dashboard |
| LightGBM | ML forecasting model |
| Scikit-learn | Model evaluation |
| Render | Cloud deployment |

---

## How to Run Locally

```bash
# Clone repository
git clone https://github.com/imakash45/rossmann-sales-forecasting.git
cd rossmann-sales-forecasting

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r dashboard/requirements.txt

# Download dataset from Kaggle
# Place CSV files in data/raw/

# Run notebooks in order
# 01_EDA.ipynb
# 02_Feature_Engineering.ipynb
# 03_Model_Training.ipynb

# Run dashboard
python dashboard/app.py
```

---

## Dataset

Download from Kaggle:
[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales)

---

## Author

**Akash Kumar**  
[GitHub](https://github.com/imakash45)

---

## Acknowledgements

- Rossmann Store Sales dataset — Kaggle
- LightGBM — Microsoft
- Plotly Dash — Plotly