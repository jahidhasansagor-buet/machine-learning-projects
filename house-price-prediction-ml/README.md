# House Price Prediction & Deployment

## Overview
A complete machine learning project that predicts US house prices based on area-level statistics.
The project covers exploratory data analysis, training and comparing multiple regression models,
and deploying the best model as an interactive web app using Gradio.

## Dataset
- **Source:** USA Housing Dataset
- **Features used:**
  - Avg. Area Income
  - Avg. Area House Age
  - Avg. Area Number of Rooms
  - Avg. Area Number of Bedrooms
  - Area Population
- **Target:** House Price
- **Total samples:** 5,000

## Model Comparison

| Model | Train R² | Test R² | Test MSE |
|-------|----------|---------|----------|
| Linear Regression | 0.9180 | 0.9180 | 1.008901e+10 |
| Ridge Regression (alpha=10) | 0.9180 | 0.9180 | 1.008965e+10 |
| Lasso Regression (alpha=100) | 0.9180 | 0.9180 | 1.008859e+10 |
| Polynomial (deg=2) + Ridge | 0.9181 |0.9179 | 1.009954e+10 |
| KNN (k=10) | 0.8987 | 0.8753 | 1.534290e+10 |
| KNN (k=5) | 0.9109 | 0.8693 | 1.607824e+10 |
| KNN (k=3) | 0.9280 | 0.8507 | 1.836970e+10 |
| KNN (k=1) | 1.0000 | 0.7767 | 2.747168e+10 |

## Final Model

**Model:** Linear Regression 
**Test R²:** 0.9180  
**Test MSE:** 1.0089009301e+10  

**Why this model?**  
Linear Regression gives almost the same performance on the training and test sets (Train R² ≈ Test R²), so it does not overfit. At the same time, its Test R² and Test MSE are as good as or better than the more complex models, while the model itself stays simple and easy to interpret.

## Web Application

Deployed using [Gradio](https://gradio.app).

### Screenshots

![Gradio Interface](screenshots/gradio_interface.png)

## Installation

```bash
git clone https://github.com/jahidhasansagor-buet/machine-learning-projects.git
cd machine-learning-projects/house-price-prediction-ml
pip install -r requirements.txt
```

## Usage

**Run EDA notebook:**
```
notebooks/1_eda.ipynb
```

**Run Training notebook:**
```
notebooks/2_training.ipynb
```

**Launch web app:**
```bash
python app.py
```

## Project Structure

```
house-price-prediction-ml/
├── data/
│   └── USA_Housing.csv
├── notebooks/
│   ├── 1_eda.ipynb
│   └── 2_training.ipynb
├── app.py
├── models/
│   └── best_model.pkl
├── screenshots/
│   └── gradio_interface.png
├── README.md
└── requirements.txt
```

## Tools and Libraries

- Python
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- Gradio
- Joblib
