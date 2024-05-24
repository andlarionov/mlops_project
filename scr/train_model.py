import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import joblib

# Открываем файл
df = pd.read_csv('datasets/df.csv')

# Разделяем признаки и целевую переменную
X = df.drop('discount_applied', axis=1)
y = df['discount_applied']

# Преобразуем целевую переменную с помощью LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Разделяем данные на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Преобразуем возрастные данные с помощью StandardScaler
scaler = StandardScaler()
X_train[['age']] = scaler.fit_transform(X_train[['age']])
X_test[['age']] = scaler.transform(X_test[['age']])

# Кодируем категориальные признаки с помощью OrdinalEncoder
ordinal = OrdinalEncoder()
X_train[['gender', 'category', 'size', 'subscription_status']] = ordinal.fit_transform(
    X_train[['gender', 'category', 'size', 'subscription_status']]
)
X_test[['gender', 'category', 'size', 'subscription_status']] = ordinal.transform(
    X_test[['gender', 'category', 'size', 'subscription_status']]
)

# Обучаем модель
model = RandomForestClassifier(
    n_estimators=200, max_depth=None, min_samples_split=10, 
    min_samples_leaf=5, bootstrap=True, class_weight='balanced', 
    random_state=42
)
model.fit(X_train, y_train)

# Предсказание на тестовом наборе
y_pred = model.predict(X_test)

# Вычисляем F1-score
f1 = f1_score(y_test, y_pred)
print(f"F1-score: {f1}")

# Сохраняем модель и предобработчики
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(ordinal, 'ordinal_encoder.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Преобразуем целевой признак в pd для сохранения в формате csv
# и последующего испоззования в тестировании
y_test = pd.DataFrame({'y_test': y_test})

# Сохраняем преобразованные данные
X_train.to_csv('datasets/X_train.csv', index=False)
X_test.to_csv('datasets/X_test.csv', index=False)
y_test.to_csv('datasets/y_test.csv', index=False)
