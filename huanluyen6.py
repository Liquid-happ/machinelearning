import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import os
import joblib
import logging

logging.basicConfig(
    filename='model_training_lstm.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

CSV_FILE = os.path.join(os.getcwd(), 'all_cities_aqi_data.csv')
MODEL_FILE = os.path.join(os.getcwd(), 'models', 'aqi_lstm_model.h5')
FEATURES_FILE = os.path.join(os.getcwd(), 'models', 'features_lstm.txt')
SCALER_FILE = os.path.join(os.getcwd(), 'models', 'scaler_lstm.pkl')
RETRAIN_FLAG = os.path.join(os.getcwd(), 'retrain_flag.txt')

def log_and_print(message):
    logging.info(message)
    print(message)

def should_retrain():
    os.makedirs('models', exist_ok=True)
    if not os.path.exists(MODEL_FILE):
        return True
    if os.path.exists(RETRAIN_FLAG):
        try:
            os.remove(RETRAIN_FLAG)
            return True
        except OSError as e:
            log_and_print(f"LỖI: {e}")
            return True
    return os.path.getmtime(CSV_FILE) > os.path.getmtime(MODEL_FILE)

def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

class AQILossLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if hasattr(self.model, 'validation_data') and self.model.validation_data is not None:
            X_val, y_val = self.model.validation_data[0], self.model.validation_data[1]
            y_pred_val = self.model.predict(X_val, verbose=0)
            mse_aqi = mean_squared_error(y_val, y_pred_val)
            log_and_print(f"Epoch {epoch + 1} - AQI Validation MSE: {mse_aqi:.4f}")

def train_model():
    log_and_print("Bắt đầu huấn luyện mô hình LSTM...")

    os.makedirs('models', exist_ok=True)

    try:
        df = pd.read_csv(CSV_FILE, encoding='utf-8-sig')
    except FileNotFoundError:
        log_and_print(f"Không tìm thấy file CSV: {CSV_FILE}")
        raise
    except Exception as e:
        log_and_print(f"Lỗi đọc CSV: {e}")
        raise

    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f%z', errors='coerce')
    if df['timestamp'].isna().any():
        log_and_print(f"Tìm thấy {df['timestamp'].isna().sum()} timestamp không hợp lệ")
        df = df.dropna(subset=['timestamp'])

    df['aqi'] = pd.to_numeric(df['aqi'], errors='coerce')
    df = df.set_index('timestamp')
    df['aqi'] = df['aqi'].interpolate(method='time')
    df = df.reset_index()

    if df['aqi'].isna().any():
        log_and_print("Cảnh báo: Vẫn còn giá trị thiếu sau khi nội suy")
        df['aqi'] = df['aqi'].fillna(df['aqi'].mean())

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

    df['aqi_mean_3h'] = df.groupby('city')['aqi'].shift(1).rolling(window=3, min_periods=1).mean()
    df['aqi_mean_3h'] = df['aqi_mean_3h'].fillna(df['aqi_mean_3h'].mean())

    df = pd.get_dummies(df, columns=['city'], drop_first=True)
    valid_cities = []
    for city in df.filter(like='city_').columns:
        city_name = city.split('city_')[1]
        if len(df[df[city] == 1]) < 24:
            log_and_print(f"Bỏ qua {city_name} vì không đủ dữ liệu (< 24 records)")
            df = df[df[city] != 1]
        else:
            valid_cities.append(city_name)
    if not valid_cities:
        log_and_print("Không đủ chuỗi dữ liệu (>= 24 records)")
        raise ValueError("Không đủ chuỗi dữ liệu (>= 24 records)")

    features = [col for col in df.columns if col in ['year', 'month', 'day', 'hour', 'day_of_week', 'is_weekend', 'sin_hour', 'cos_hour', 'aqi_mean_3h'] or col.startswith('city_')]
    X = df[features].dropna()
    y = df.loc[X.index, 'aqi'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_FILE)
    log_and_print(f"Lưu scaler vào {SCALER_FILE}")

    time_steps = 24
    X_seq, y_seq = create_sequences(X_scaled, y, time_steps)

    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, input_shape=(time_steps, X_train.shape[2]))),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)  # Chỉ dự đoán AQI
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
    log_and_print("Khởi tạo mô hình huấn luyện")

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_loss')
    aqi_logger = AQILossLogger()

    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, checkpoint, aqi_logger],
        verbose=1
    )

    y_pred = model.predict(X_test, verbose=0)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    log_and_print(f"AQI - Mean Squared Error: {mse:.2f}, R² Score: {r2:.2f},MAE: {mae:.2f}")

    model.save(MODEL_FILE)
    log_and_print(f"Lưu mô hình vào {MODEL_FILE}")
    with open(FEATURES_FILE, 'w', encoding='utf-8') as f:
        f.write(','.join(features))
    log_and_print(f"Lưu các đặc trưng vào {FEATURES_FILE}")

if __name__ == "__main__":
    if should_retrain():
        train_model()
    else:
        log_and_print("Mô hình đã được cập nhật.")
