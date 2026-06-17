# Retail-Sales-Forecasting
An end-to-end machine learning project that predicts future sales for specific items across different stores. Built with **LightGBM** and deployed as an interactive web application using **Streamlit**
## 🚀 Live Demo
LINK######################

## 🛠️ Tech Stack
* **Data Processing:** Python, Pandas, NumPy
* **Machine Learning:** Scikit-Learn, LightGBM (Gradient Boosting)
* **Web Framework:** Streamlit
* **Model Persistence:** Joblib

## 📈 Improving Model Accuracy: The Journey
The core challenge of this project was capturing the complex, non-linear relationships in time-series retail data. The model's accuracy was iteratively improved through the following phases:

### Phase 1: Baseline Model
* **Approach:** Trained an initial gradient boosting regressor on the raw dataset using basic identifiers (`store_id`, `item_id`, `price`, `promo`).
* **Outcome:** The model struggled to capture seasonality and purchasing trends, leading to higher prediction errors.

### Phase 2: Temporal Feature Engineering
* **Approach:** Extracted specific time-bound components from the raw `date` column.
* **Features Added:** `month`, `day_of_week`, and a binary `is_weekend` indicator.
* **Outcome:** Accuracy saw a significant bump as the model could now differentiate between weekday slumps and weekend demand spikes, as well as seasonal monthly shifts.

### Phase 3: Autoregressive Lag Features
* **Approach:** Recognizing that recent past performance is the strongest predictor of near-future performance, sliding window metrics were introduced.
* **Features Added:** `sales_lag_1` (yesterday's sales) and `sales_lag_7` (sales from the exact same day last week).
* **Outcome:** This provided the most substantial improvement to the $R^2$ score and MAE. The model successfully learned to adjust baseline predictions based on immediate real-world momentum.

### Phase 4: Model Optimization & Deployment
* **Approach:** Fine-tuned the LightGBM hyperparameters and aligned the feature matrices using one-hot encoding (`pd.get_dummies()`).
* **Deployment:** Exported the optimized model and exact feature columns to ensure flawless translation of user inputs in the Streamlit production environment.

## 💻 Running the App Locally

1. Clone this repository to your local machine.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
