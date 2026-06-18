import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

st.title("Анализ цен смартфонов")

df = pd.read_csv("smartphones.xls", sep=",", encoding="utf-8-sig")

st.subheader("Датасет")
st.dataframe(df.head())

data = df.copy()

data = data.replace([np.inf, -np.inf], np.nan)

numeric_target_columns = data.select_dtypes(
    include=[np.number]
).columns.tolist()

target_column = st.selectbox(
    "Выберите целевой признак",
    numeric_target_columns
)

X_raw = data.drop(columns=[target_column])
y_raw = data[target_column]

X = X_raw.copy()

numeric_cols = X.select_dtypes(include=[np.number]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

if len(numeric_cols) > 0:
    num_imputer = SimpleImputer(strategy="median")
    X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

if len(categorical_cols) > 0:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

    le = LabelEncoder()
    for col in categorical_cols:
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])

for col in X.columns:
    if X[col].dtype == "object":
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

if X.isnull().any().any():
    X = X.fillna(0)

test_size = st.slider(
    "Размер тестовой выборки",
    0.1,
    0.5,
    0.2
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_raw,
    test_size=test_size,
    random_state=42
)

model_name = st.selectbox(
    "Выберите модель",
    ["Linear Regression", "Decision Tree", "Random Forest", "MLP"]
)

max_depth = None
n_estimators = 100
hidden_layer_size = 100

if model_name == "Decision Tree":
    max_depth = st.slider("max_depth", 1, 20, 5)
elif model_name == "Random Forest":
    n_estimators = st.slider("n_estimators", 10, 500, 100)
elif model_name == "MLP":
    hidden_layer_size = st.slider("neurons", 10, 300, 100)

if st.button("Обучить модель"):

    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Decision Tree":
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=42
        )
    elif model_name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42
        )
    elif model_name == "MLP":
        model = MLPRegressor(
            hidden_layer_sizes=(hidden_layer_size,),
            max_iter=1000,
            random_state=42
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Метрики качества")
    st.write(f"MAE: {mae:.4f}")
    st.write(f"MSE: {mse:.4f}")
    st.write(f"RMSE: {rmse:.4f}")
    st.write(f"R²: {r2:.4f}")

    st.subheader("Сравнение")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test.values, label="Фактические")
    ax.plot(y_pred, label="Прогноз")
    ax.set_xlabel("Наблюдение")
    ax.set_ylabel(target_column)
    ax.legend()
    st.pyplot(fig)
