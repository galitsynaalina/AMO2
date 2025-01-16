import pandas as pd
import numpy as np
import requests
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# 1. Сбор данных
def download_data(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"Данные загружены и сохранены как {filename}")


# 2. Предобработка данных
def preprocess_data(filepath):
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data = pd.read_csv(filepath, names=column_names)
    data = data.dropna()  # Удаляем пропуски
    X = data.drop('Outcome', axis=1)  # 'Outcome' - целевой признак
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Данные успешно разделены на обучающую и тестовую выборки")
    return X_train, X_test, y_train, y_test


# 3. Обучение модели
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Модель обучена")
    return model


# 4. Оценка модели
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print(f"Точность модели: {accuracy}")
    print("Отчет классификации:\n", report)


# 5. Сохранение модели
def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Модель сохранена в {filename}")


# Основной конвейер
def main():
    data_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
    data_file = 'data.csv'
    model_file = 'model.pkl'

    download_data(data_url, data_file)
    X_train, X_test, y_train, y_test = preprocess_data(data_file)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, model_file)


if __name__ == "__main__":
    main()