🚗 Car Price Prediction using Machine Learning + Streamlit

This project predicts the **resale price of a car** based on its features such as present price, kilometers driven, fuel type, seller type, transmission, number of owners, and age.
The machine learning model is trained using **XGBoost**, which provided the best performance among multiple regression algorithms.


📂 Project Structure

car-price-prediction/
│
├── data/
│   └── cardata.xls              \# Dataset
│
├── models/
│   ├── cardata                  \# Trained joblib model
│   └── xgb\_model.json           \# Trained XGBoost model
│
├── train.py                     \# Script to train & save the model
├── streamlit\_app.py             \# Streamlit prediction app
├── requirements.txt             \# Project dependencies
└── README.md                    \# Project documentation


⚙️ Installation & Setup

1.  **Clone this repository**
    bash
    git clone [https://github.com/jairamgolla07/car-price-prediction-project/]
    cd car-price-prediction
    

2.  **Create a virtual environment** (optional but recommended)
    ```bash
    # Create the environment
    python -m venv venv

    # Activate it
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏋️ Train the Model

Run the training script to process data, train models, evaluate their performance, and save the best one (XGBoost):

```bash
python train.py
````

This will generate two model files in the `models/` directory:

  * `models/cardata` (joblib saved model)
  * `models/xgb_model.json` (XGBoost saved model)

-----

## ▶️ Run the Streamlit App Locally

Launch the web application from your terminal:

```bash
streamlit run streamlit_app.py
```

## 📊 Algorithms Used

  * Linear Regression
  * Random Forest Regressor
  * Gradient Boosting Regressor
  * **XGBoost Regressor** ✅ (Best performing, used in the final app)

-----

## 📌 Requirements

The `requirements.txt` file contains:

```
streamlit
pandas
scikit-learn
xgboost
joblib
openpyxl
```



## 🙌 Author

Developed by **Jai Ram**

  * GitHub: [@jairamgolla07]

<!-- end list -->

```
```
