import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Загрузить модель и объекты предобработки
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
ordinal = joblib.load("ordinal_encoder.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("Discount Application Prediction")

# Описание приложения
st.write(
    """
Это приложение предсказывает, будет ли применена скидка на основе предоставленных данных. 
Пожалуйста, введите данные ниже.
"""
)

# Интерактивные элементы управления
age = st.slider("Age", 14, 100, 43)
gender = st.selectbox("Gender", ["Male", "Female"])
category = st.selectbox(
    "Category", ["Clothing", "Accessories", "Footwear", "Outerwear"]
)
size = st.selectbox("Size", ["M", "L", "S", "XL"])
subscription_status = st.selectbox("Subscription Status", ["Yes", "No"])

# Подготовить входные данные в формате, ожидаемом моделью
input_data = pd.DataFrame(
    {
        "age": [age],
        "gender": [gender],
        "category": [category],
        "size": [size],
        "subscription_status": [subscription_status],
    }
)

# Преобразовать возраст с помощью scaler
input_data["age"] = scaler.transform(input_data[["age"]])

# Преобразовать категориальные признаки в числовые значения с помощью ordinal encoder
input_data[["gender", "category", "size", "subscription_status"]] = ordinal.transform(
    input_data[["gender", "category", "size", "subscription_status"]]
)

# Предсказание
prediction = model.predict(input_data)

# Конвертировать предсказание обратно в исходные метки
prediction_label = label_encoder.inverse_transform(prediction)

# Отобразить предсказание
st.write("Predicted discount application:", prediction_label[0])
