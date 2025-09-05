# House Price Prediction (Flask + ML)

End-to-end project:
- EDA & data visualization
- Train & compare **Linear Regression vs XGBoost**
- Auto-select **best model**, save `models/model.pkl`
- Flask web app for price prediction

## Structure
- `train.py` – trains models, saves best to `models/`,– saves plots to `images/`
- `app.py` – Flask app (loads best model)
- `data/train.csv` – Kaggle House Prices dataset (place here)

## Quickstart
```bash
pip install -r requirements.txt

# 1) Train + select best model
python train.py

# 2) Run web app
python app.py
# open http://127.0.0.1:5000
