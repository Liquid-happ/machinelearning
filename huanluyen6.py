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

# Configure logging
logging.basicConfig(
    filename='model_training_lstm.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# File paths
CSV_FILE = os.path.join(os.getcwd(), 'all_cities_aqi_data.csv')
MODEL_FILE = os.path.join(os.getcwd(), 'models', 'aqi_multioutput_lstm_model.h5')
FEATURES_FILE = os.path.join(os.getcwd(), 'models', 'features_lstm.txt')
SCALER_FILE = os.path.join(os.getcwd(), 'models', 'scaler_lstm.pkl')
RETRAIN_FLAG = os.path.join(os.getcwd(), 'retrain_flag.txt')

def should_retrain():
    """Check if model needs retraining"""
    if not os.path.exists(MODEL_FILE):
        return True
    if os.path.exists(RETRAIN_FLAG):
        os.remove(RETRAIN_FLAG)
        return True
    data_mtime = os.path.getmtime(CSV_FILE)
    model_mtime = os.path.getmtime(MODEL_FILE)
    return data_mtime > model_mtime

def create_sequences(X, y, time_steps=24):
    """Create sequences for LSTM input"""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def train_model():
    """Train multi-output LSTM model for AQI, wind speed, and humidity"""
    logging.info("Starting LSTM model training...")
    print("Starting LSTM model training...")

    # Read data with UTF-8 encoding
    df = pd.read_csv(CSV_FILE, encoding='utf-8')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', errors='coerce')

    # Handle missing data
    df[['aqi', 'wind_speed', 'humidity']] = df[['aqi', 'wind_speed', 'humidity']].interpolate(method='linear')

    # Extract time-based features
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Normalize wind_speed and humidity
    df['wind_speed'] = df['wind_speed'].str.replace(' km/h', '').astype(float)
    df['humidity'] = df['humidity'].str.replace('%', '').astype(float)

    # Add historical features (3-hour moving average)
    for col in ['aqi', 'wind_speed', 'humidity']:
        df[f'{col}_mean_3h'] = df.groupby('city')[col].shift(1).rolling(window=3).mean()
    df[['aqi_mean_3h', 'wind_speed_mean_3h', 'humidity_mean_3h']] = df[['aqi_mean_3h', 'wind_speed_mean_3h', 'humidity_mean_3h']].fillna(df[['aqi_mean_3h', 'wind_speed_mean_3h', 'humidity_mean_3h']].mean())

    # One-hot encode city
    df = pd.get_dummies(df, columns=['city'], drop_first=True)

    # Check data sufficiency per city
    for city in df.filter(like='city_').columns:
        city_name = city.split('city_')[1]
        if len(df[df[city] == 1]) < 24:
            raise ValueError(f"Insufficient data for city {city_name} (need at least 24 records)")

    # Features and labels
    features = ['year', 'month', 'day', 'hour', 'day_of_week', 'is_weekend', 'sin_hour', 'cos_hour',
                'aqi_mean_3h', 'wind_speed_mean_3h', 'humidity_mean_3h'] + \
               [col for col in df.columns if col.startswith('city_')]
    X = df[features].dropna()
    y = df.loc[X.index, ['aqi', 'wind_speed', 'humidity']].values

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_FILE)
    logging.info(f"Saved scaler to {SCALER_FILE}")

    # Create sequences for LSTM
    time_steps = 24
    X_seq, y_seq = create_sequences(X_scaled, y, time_steps)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    # Build LSTM model
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, input_shape=(time_steps, X_train.shape[2]))),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(3)  # 3 outputs: aqi, wind_speed, humidity
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
    logging.info("Created model architecture")

    # Train model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_loss')
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    # Evaluate model
    y_pred = model.predict(X_test)
    for i, target in enumerate(['AQI', 'Wind Speed', 'Humidity']):
        mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        logging.info(f"{target} - Mean Squared Error: {mse:.2f}, R² Score: {r2:.2f}")
        print(f"{target} - Mean Squared Error: {mse:.2f}, R² Score: {r2:.2f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    model.save(MODEL_FILE)
    logging.info(f"Saved model to {MODEL_FILE}")
    print(f"Saved model to {MODEL_FILE}")

    # Save feature list
    with open(FEATURES_FILE, 'w', encoding='utf-8') as f:
        f.write(','.join(features))
    logging.info(f"Saved features to {FEATURES_FILE}")
    print(f"Saved features to {FEATURES_FILE}")

if __name__ == "__main__":
    if should_retrain():
        train_model()
    else:
        print("No retraining needed. Model is up to date.")