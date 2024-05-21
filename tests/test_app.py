import pytest
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import numpy as np

y_test = pd.read_csv('/datasets/y_test.csv')['y_test'].values
X_test = pd.read_csv('/datasets/X_test.csv')

@pytest.fixture(scope='module')
def load_model_and_preprocessors():
    model = joblib.load('../scr/model.pkl')
    scaler = joblib.load('../scr/scaler.pkl')
    ordinal = joblib.load('../scr/ordinal_encoder.pkl')
    label_encoder = joblib.load('../scr/label_encoder.pkl')
    return model, scaler, ordinal, label_encoder

@pytest.fixture
def input_data():
    return pd.DataFrame({
        'age': [25],
        'gender': ['Male'],
        'category': ['Clothing'],
        'size': ['M'],
        'subscription_status': ['Yes']
    })

def test_scaler(load_model_and_preprocessors, input_data):
    model, scaler, ordinal, label_encoder = load_model_and_preprocessors
    # Проверка масштабирования возраста
    age_scaled = scaler.transform(input_data[['age']])
    assert age_scaled.shape == (1, 1)

def test_ordinal_encoder(load_model_and_preprocessors, input_data):
    model, scaler, ordinal, label_encoder = load_model_and_preprocessors
    # Проверка кодирования категориальных признаков
    encoded_data = ordinal.transform(input_data[['gender', 'category', 'size', 'subscription_status']])
    assert encoded_data.shape == (1, 4)

def test_prediction(load_model_and_preprocessors, input_data):
    model, scaler, ordinal, label_encoder = load_model_and_preprocessors
    input_data_copy = input_data.copy()
    input_data_copy['age'] = scaler.transform(input_data_copy[['age']])
    input_data_copy[['gender', 'category', 'size', 'subscription_status']] = ordinal.transform(
        input_data_copy[['gender', 'category', 'size', 'subscription_status']]
    )

    prediction = model.predict(input_data_copy)
    assert prediction[0] in [0, 1]


    # Предсказания
    y_pred = model.predict(X_test)

    # Вычисление F1-score
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred)
    assert f1 > 0.7