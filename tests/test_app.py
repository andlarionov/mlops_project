import os # use only on the OS Windows
import pytest
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import numpy as np

y_test = pd.read_csv('datasets/y_test.csv')['y_test'].values  # for Linux
X_test = pd.read_csv('datasets/X_test.csv')   # for Linux

# y_test = pd.read_csv(os.pardir+'\\datasets\\y_test.csv')['y_test'].values   # for Windows
# X_test = pd.read_csv(os.pardir+'\\datasets\\X_test.csv')    # for Windows

"""Paths for Linux"""
@pytest.fixture(scope='module')
def load_model_and_preprocessors():
    model = joblib.load('scr/model.pkl')
    scaler = joblib.load('scr/scaler.pkl')
    ordinal = joblib.load('scr/ordinal_encoder.pkl')
    label_encoder = joblib.load('scr/label_encoder.pkl')
    return model, scaler, ordinal, label_encoder

# """Paths for Windows"""
# @pytest.fixture(scope='module')
# def load_model_and_preprocessors():
#     model = joblib.load(os.pardir+'\\scr\\model.pkl')
#     scaler = joblib.load(os.pardir+'\\scr\\scaler.pkl')
#     ordinal = joblib.load(os.pardir+'\\scr\\ordinal_encoder.pkl')
#     label_encoder = joblib.load(os.pardir+'\\scr\\label_encoder.pkl')
#     return model, scaler, ordinal, label_encoder

@pytest.fixture
def input_data():
    return pd.DataFrame({
        'age': [25],
        'gender': ['Male'],
        'category': ['Clothing'],
        'size': ['M'],
        'subscription_status': ['Yes']
    })

@pytest.fixture
def input_dataset():
    df = pd.read_csv('datasets/df.csv')    # for Linux
    # df = pd.read_csv(os.pardir + '\\datasets\\df.csv')  # for Windows
    return df

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


def test_data_age(input_dataset):
    count_age = 0
    for age in input_dataset['age']:
        count_age += 1
        assert isinstance(age, int), f"Value of [age] in string {count_age} is empty or is not integer, value: {age}"


def test_data_gender(input_dataset):
    count_gender = 0
    gender_list = ['Male', 'Female']
    for gender in input_dataset['gender']:
        count_gender += 1
        assert gender in gender_list, \
            f"Value of [gender] in string {count_gender} is not 'Male' or 'Female', value: {gender}"


def test_data_category(input_dataset):
    count_category = 0
    category_list = ['Clothing', 'Footwear', 'Outerwear', 'Accessories']
    for category in input_dataset['category']:
        count_category += 1
        assert category in category_list,\
            f"Value of [category] in string {count_category} is not {category_list}, value: {category}"


def test_data_size(input_dataset):
    count_size = 0
    size_list = ['XS', 'S', 'L', 'M', 'XL', 'XXL', 'XXXL']
    for size in input_dataset['size']:
        count_size += 1
        assert size in size_list,\
            f"Value of [category] in string {count_size} is not {size_list}, value: {size}"


def test_data_subscription_status(input_dataset):
    count_subscription_status = 0
    subscription_list = ['Yes', 'No']
    for subscription_status in input_dataset['subscription_status']:
        count_subscription_status += 1
        assert subscription_status in subscription_list,\
            (f"Value of [subscription_status] in string {count_subscription_status} is not {subscription_list}, "
             f"value: {subscription_status}")


def test_data_discount_applied(input_dataset):
    count_discount_applied = 0
    discount_applied_list = ['Yes', 'No']
    for discount_applied in input_dataset['discount_applied']:
        count_discount_applied += 1
        assert discount_applied in discount_applied_list, \
            (f"Value of [subscription_status] in string {count_discount_applied} is not {discount_applied_list}, "
             f"value: {discount_applied}")
