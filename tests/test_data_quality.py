import os # use only on the OS Windows
import pytest
import pandas as pd


@pytest.fixture
def input_dataset():
    df = pd.read_csv('datasets/df.csv')    # for Linux
    # df = pd.read_csv(os.pardir + '\\datasets\\df.csv')  # for Windows
    return df


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
