import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import joblib
import logging

# Cấu hình logging
logging.basicConfig(
    filename='model_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# Đường dẫn đến file CSV và mô hình
CSV_FILE = os.path.join(os.getcwd(), 'all_cities_aqi_data.csv')
MODEL_FILE = os.path.join(os.getcwd(), 'models', 'aqi_multioutput_xgboost_model.pkl')
FEATURES_FILE = os.path.join(os.getcwd(), 'models', 'features.txt')
RETRAIN_FLAG = os.path.join(os.getcwd(), 'retrain_flag.txt')

def should_retrain():
    """Check if retraining is needed based on file modification or flag"""
    if not os.path.exists(MODEL_FILE):
        return True
    if os.path.exists(RETRAIN_FLAG):
        os.remove(RETRAIN_FLAG)
        return True
    data_mtime = os.path.getmtime(CSV_FILE)
    model_mtime = os.path.getmtime(MODEL_FILE)
    return data_mtime > model_mtime

def train_model():
    """Train the multi-output model for AQI, wind speed, and humidity"""
    logging.info("Starting model training...")
    print("Starting model training...")

    # Đọc dữ liệu với mã hóa utf-8
    df = pd.read_csv(CSV_FILE, encoding='utf-8')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Trích xuất đặc trưng thời gian
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # Chuẩn hóa wind_speed và humidity
    df['wind_speed'] = df['wind_speed'].str.replace(' km/h', '').astype(float)
    df['humidity'] = df['humidity'].str.replace('%', '').astype(float)

    # Chuyển city thành one-hot encoding
    df = pd.get_dummies(df, columns=['city'], drop_first=True)

    # Đặc trưng và nhãn
    features = ['year', 'month', 'day', 'hour', 'day_of_week'] + \
               [col for col in df.columns if col.startswith('city_')]
    X = df[features]
    y = df[['aqi', 'wind_speed', 'humidity']]

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tối ưu hóa mô hình XGBoost với GridSearchCV
    xgb_model = XGBRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9]
    }
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5,
                               scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Lấy mô hình tốt nhất
    best_model = grid_search.best_estimator_
    logging.info(f"Best parameters: {grid_search.best_params_}")
    print(f"Best parameters: {grid_search.best_params_}")

    # Đánh giá mô hình
    y_pred = best_model.predict(X_test)
    for i, target in enumerate(['AQI', 'Wind Speed', 'Humidity']):
        mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        logging.info(f"{target} - Mean Squared Error: {mse:.2f}, R² Score: {r2:.2f}")
        print(f"{target} - Mean Squared Error: {mse:.2f}, R² Score: {r2:.2f}")

    # Lưu mô hình
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, MODEL_FILE)
    logging.info(f"Model saved to {MODEL_FILE}")
    print(f"Model saved to {MODEL_FILE}")

    # Lưu danh sách đặc trưng
    with open(FEATURES_FILE, 'w', encoding='utf-8') as f:
        f.write(','.join(features))
    logging.info(f"Features saved to {FEATURES_FILE}")
    print(f"Features saved to {FEATURES_FILE}")

if __name__ == "__main__":
    if should_retrain():
        train_model()
    else:
        print("No retraining needed. Model is up-to-date.")