import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import mean_squared_error, r2_score
import os
import joblib
import logging

# Cấu hình logging
logging.basicConfig(
    filename='model_training_lstm.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# Đường dẫn file
CSV_FILE = os.path.join(os.getcwd(), 'all_cities_aqi_data.csv')
MODEL_FILE = os.path.join(os.getcwd(), 'models', 'aqi_multioutput_lstm_model.h5')
FEATURES_FILE = os.path.join(os.getcwd(), 'models', 'features_lstm.txt')
SCALER_FILE = os.path.join(os.getcwd(), 'models', 'scaler_lstm.pkl')
RETRAIN_FLAG = os.path.join(os.getcwd(), 'retrain_flag.txt')

def log_and_print(message):
    """Log and print message for consistency"""
    logging.info(message)
    print(message)

def should_retrain():
    """Check if model needs retraining"""
    os.makedirs('models', exist_ok=True)
    if not os.path.exists(MODEL_FILE):
        return True
    if os.path.exists(RETRAIN_FLAG):
        try:
            os.remove(RETRAIN_FLAG)
            return True
        except OSError as e:
            log_and_print(f"Error removing retrain flag: {e}")
            return True
    return os.path.getmtime(CSV_FILE) > os.path.getmtime(MODEL_FILE)

def create_sequences(X, y, time_steps=24):
    """Create sequences for LSTM input"""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def train_model():
    """Train multi-output LSTM model for AQI, wind speed, and humidity"""
    log_and_print("Starting LSTM model training...")

    # Tạo thư mục models
    os.makedirs('models', exist_ok=True)

    # Đọc và xử lý dữ liệu
    try:
        df = pd.read_csv(CSV_FILE, encoding='utf-8-sig')
    except FileNotFoundError:
        log_and_print(f"CSV file not found: {CSV_FILE}")
        raise
    except Exception as e:
        log_and_print(f"Error reading CSV: {e}")
        raise

    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f%z', errors='coerce')
    if df['timestamp'].isna().any():
        log_and_print(f"Found {df['timestamp'].isna().sum()} invalid timestamps")
        df = df.dropna(subset=['timestamp'])

    # Chuyển đổi kiểu dữ liệu
    df['aqi'] = pd.to_numeric(df['aqi'], errors='coerce')
    df['wind_speed'] = df['wind_speed'].str.replace(' km/h', '', regex=False).astype(float, errors='ignore')
    df['humidity'] = df['humidity'].str.replace('%', '', regex=False).astype(float, errors='ignore')

    # Nội suy theo thời gian
    df = df.set_index('timestamp')
    df[['aqi', 'wind_speed', 'humidity']] = df[['aqi', 'wind_speed', 'humidity']].infer_objects(copy=False).interpolate(method='time')
    df = df.reset_index()

    # Kiểm tra dữ liệu thiếu
    if df[['aqi', 'wind_speed', 'humidity']].isna().any().any():
        log_and_print("Warning: Missing values remain after interpolation")
        df[['aqi', 'wind_speed', 'humidity']] = df[['aqi', 'wind_speed', 'humidity']].fillna(df[['aqi', 'wind_speed', 'humidity']].mean())

    # Trích xuất đặc trưng
    df = df.assign(
        year=df['timestamp'].dt.year,
        month=df['timestamp'].dt.month,
        day=df['timestamp'].dt.day,
        hour=df['timestamp'].dt.hour,
        day_of_week=df['timestamp'].dt.dayofweek,
        is_weekend=df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int),
        sin_hour=np.sin(2 * np.pi * df['timestamp'].dt.hour / 24),
        cos_hour=np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
    )

    # Thêm trung bình động 3 giờ
    for col in ['aqi', 'wind_speed', 'humidity']:
        df[f'{col}_mean_3h'] = df.groupby('city')[col].shift(1).rolling(window=3, min_periods=1).mean()
    df[['aqi_mean_3h', 'wind_speed_mean_3h', 'humidity_mean_3h']] = df[['aqi_mean_3h', 'wind_speed_mean_3h', 'humidity_mean_3h']].fillna(df[['aqi_mean_3h', 'wind_speed_mean_3h', 'humidity_mean_3h']].mean())

    # One-hot encoding
    df = pd.get_dummies(df, columns=['city'], drop_first=True)
    for city in df.filter(like='city_').columns:
        city_name = city.split('city_')[1]
        if len(df[df[city] == 1]) < 24:
            raise ValueError(f"Insufficient data for city {city_name} (need at least 24 records)")

    # Đặc trưng và nhãn
    features = [col for col in df.columns if col in ['year', 'month', 'day', 'hour', 'day_of_week', 'is_weekend', 'sin_hour', 'cos_hour', 'aqi_mean_3h', 'wind_speed_mean_3h', 'humidity_mean_3h'] or col.startswith('city_')]
    X = df[features].dropna()
    y = df.loc[X.index, ['aqi', 'wind_speed', 'humidity']].values

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_FILE)
    log_and_print(f"Saved scaler to {SCALER_FILE}")

    # Tạo chuỗi cho LSTM
    time_steps = 24
    X_seq, y_seq = create_sequences(X_scaled, y, time_steps)

    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    # Xây dựng mô hình LSTM
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, input_shape=(time_steps, X_train.shape[2]))),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(3)  # 3 outputs: aqi, wind_speed, humidity
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
    log_and_print("Created model architecture")

    # Huấn luyện mô hình
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_loss')
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    # Đánh giá mô hình
    y_pred = model.predict(X_test, verbose=0)
    for i, target in enumerate(['AQI', 'Wind Speed', 'Humidity']):
        mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        log_and_print(f"{target} - Mean Squared Error: {mse:.2f}, R² Score: {r2:.2f}")

    # Lưu mô hình và đặc trưng
    model.save(MODEL_FILE)
    log_and_print(f"Saved model to {MODEL_FILE}")
    with open(FEATURES_FILE, 'w', encoding='utf-8') as f:
        f.write(','.join(features))
    log_and_print(f"Saved features to {FEATURES_FILE}")

if __name__ == "__main__":
    if should_retrain():
        train_model()
    else:
        log_and_print("No retraining needed. Model is up to date.")
