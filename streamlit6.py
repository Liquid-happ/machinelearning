import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.express as px
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os
import time

# Set page config as the first Streamlit command
st.set_page_config(page_title="Dự đoán AQI Việt Nam", layout="wide")

# File paths
CSV_FILE = os.path.join(os.getcwd(), 'all_cities_aqi_data.csv')
MODEL_FILE = os.path.join(os.getcwd(), 'models', 'aqi_multioutput_lstm_model.h5')
FEATURES_FILE = os.path.join(os.getcwd(), 'models', 'features_lstm.txt')
SCALER_FILE = os.path.join(os.getcwd(), 'models', 'scaler_lstm.pkl')
RETRAIN_FLAG = os.path.join(os.getcwd(), 'retrain_flag.txt')

# Load model, scaler, and features
@st.cache_resource
def load_model_and_features():
    try:
        model = load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        with open(FEATURES_FILE, 'r', encoding='utf-8') as f:
            features = f.read().strip().split(',')
        return model, scaler, features
    except FileNotFoundError:
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

model, scaler, features = load_model_and_features()
if model is None or scaler is None or features is None:
    if not st.session_state.get('error_displayed', False):
        st.error("Model, scaler, or features file not found. Please train the model first.")
        st.session_state['error_displayed'] = True
    st.stop()

# Load historical data
try:
    df = pd.read_csv(CSV_FILE, encoding='utf-8')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Preprocess wind_speed and humidity to ensure numeric types
    df['wind_speed'] = df['wind_speed'].str.replace(' km/h', '', regex=False).astype(float)
    df['humidity'] = df['humidity'].str.replace('%', '', regex=False).astype(float)
    df['aqi'] = df['aqi'].astype(float)  # Ensure aqi is also float
except FileNotFoundError:
    st.error("Data file not found. Please run the data collection script first.")
    st.stop()
except ValueError as e:
    st.error(f"Error processing data: {str(e)}. Check 'aqi', 'wind_speed', and 'humidity' columns for non-numeric values.")
    st.stop()

# Get Vietnam time
def get_vietnam_time():
    return datetime.now(ZoneInfo("Asia/Bangkok"))

# Tailwind to Plotly color mapping
TAILWIND_TO_PLOTLY_COLORS = {
    'green-500': '#22c55e',
    'yellow-500': '#eab308',
    'orange-500': '#f97316',
    'red-500': '#ef4444',
    'purple-500': '#a855f7',
    'maroon-500': '#7f1d1d'
}

# Determine AQI category
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Tốt", "bg-green-500", "Không khí sạch, không gây nguy hiểm."
    elif aqi <= 100:
        return "Trung bình", "bg-yellow-500", "Chất lượng không khí chấp nhận được, nhưng một số chất ô nhiễm có thể ảnh hưởng đến nhóm nhạy cảm."
    elif aqi <= 150:
        return "Không tốt cho nhóm nhạy cảm", "bg-orange-500", "Nhóm nhạy cảm có thể gặp vấn đề sức khỏe."
    elif aqi <= 200:
        return "Có hại", "bg-red-500", "Mọi người bắt đầu cảm nhận được ảnh hưởng sức khỏe."
    elif aqi <= 300:
        return "Rất có hại", "bg-purple-500", "Cảnh báo sức khỏe: mọi người bị ảnh hưởng nghiêm trọng hơn."
    else:
        return "Nguy hiểm", "bg-maroon-500", "Cảnh báo sức khỏe khẩn cấp: toàn bộ dân số có nguy cơ bị ảnh hưởng."

# Create sequence for LSTM prediction
def create_sequence_for_prediction(city, future_datetime, df, features, scaler, time_steps=24, forecast_hours=6):
    city_data = df[df['city'] == city].sort_values('timestamp')
    if len(city_data) < time_steps:
        st.error(f"Insufficient historical data for {city} (need at least {time_steps} records).")
        return None

    # Extract time-based features for historical data
    city_data['year'] = city_data['timestamp'].dt.year
    city_data['month'] = city_data['timestamp'].dt.month
    city_data['day'] = city_data['timestamp'].dt.day
    city_data['hour'] = city_data['timestamp'].dt.hour
    city_data['day_of_week'] = city_data['timestamp'].dt.dayofweek
    city_data['is_weekend'] = city_data['day_of_week'].isin([5, 6]).astype(int)
    city_data['sin_hour'] = np.sin(2 * np.pi * city_data['hour'] / 24)
    city_data['cos_hour'] = np.cos(2 * np.pi * city_data['hour'] / 24)
    city_data['wind_speed'] = city_data['wind_speed'].astype(float)
    city_data['humidity'] = city_data['humidity'].astype(float)
    for col in ['aqi', 'wind_speed', 'humidity']:
        city_data[f'{col}_mean_3h'] = city_data[col].shift(1).rolling(window=3).mean()

    # Prepare historical sequence
    city_data = pd.get_dummies(city_data, columns=['city'], drop_first=True)
    for col in [f for f in features if f.startswith('city_')]:
        if col not in city_data.columns:
            city_data[col] = 0

    recent_data = city_data[features].tail(time_steps).values
    recent_data_scaled = scaler.transform(recent_data)

    # Predict multiple future steps
    predictions = []
    current_sequence = recent_data_scaled.copy()
    for h in range(forecast_hours):
        sequence = np.expand_dims(current_sequence, axis=0)
        pred = model.predict(sequence, verbose=0)[0]
        predictions.append(pred)

        # Update sequence with new prediction
        new_data = {
            'year': future_datetime.year,
            'month': future_datetime.month,
            'day': future_datetime.day,
            'hour': future_datetime.hour,
            'day_of_week': future_datetime.weekday(),
            'is_weekend': 1 if future_datetime.weekday() >= 5 else 0,
            'sin_hour': np.sin(2 * np.pi * future_datetime.hour / 24),
            'cos_hour': np.cos(2 * np.pi * future_datetime.hour / 24),
            'aqi_mean_3h': city_data['aqi'].tail(3).mean() if h == 0 else np.mean([p[0] for p in predictions[-3:]] if len(predictions) >= 3 else predictions[0][0]),
            'wind_speed_mean_3h': city_data['wind_speed'].tail(3).mean() if h == 0 else np.mean([p[1] for p in predictions[-3:]] if len(predictions) >= 3 else predictions[0][1]),
            'humidity_mean_3h': city_data['humidity'].tail(3).mean() if h == 0 else np.mean([p[2] for p in predictions[-3:]] if len(predictions) >= 3 else predictions[0][2])
        }
        for city_col in [col for col in features if col.startswith('city_')]:
            city_name = city_col.replace('city_', '')
            new_data[city_col] = 1 if city_name == city else 0

        new_data_scaled = scaler.transform(pd.DataFrame([new_data], columns=features))
        current_sequence = np.vstack((current_sequence[1:], new_data_scaled))
        future_datetime += timedelta(hours=1)

    return predictions

# Initialize session state
if 'selected_city' not in st.session_state:
    st.session_state.selected_city = 'Hà Nội'
if 'selected_city_pred' not in st.session_state:
    st.session_state.selected_city_pred = 'Hà Nội'
if 'future_date' not in st.session_state:
    st.session_state.future_date = datetime.today().date()
if 'future_time' not in st.session_state:
    current_time = get_vietnam_time()
    rounded_minutes = (current_time.minute // 30) * 30
    st.session_state.future_time = current_time.replace(minute=rounded_minutes, second=0, microsecond=0).time()
if 'error_displayed' not in st.session_state:
    st.session_state.error_displayed = False

# Apply Tailwind CSS
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        body { font-family: Arial, sans-serif; }
        .aqi-gauge { text-align: center; font-size: 3rem; font-weight: bold; padding: 1rem; border-radius: 0.5rem; color: white; }
        .sidebar .sidebar-content { background-color: #f8f9fa; }
        .prediction-box { background-color: #f0f4f8; padding: 1.5rem; border-radius: 0.5rem; }
    </style>
""", unsafe_allow_html=True)

# Main title
st.markdown("""
    <div class="text-center py-6">
        <h1 class="text-4xl font-bold text-gray-800">Dự đoán Chỉ số Chất lượng Không khí (AQI)</h1>
        <p class="text-lg text-gray-600">Theo dõi và dự đoán chất lượng không khí, tốc độ gió, và độ ẩm tại các thành phố lớn ở Việt Nam</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h2 class='text-2xl font-semibold text-gray-800'>Lịch sử AQI</h2>", unsafe_allow_html=True)
    cities = ['Hà Nội', 'Hồ Chí Minh', 'Đà Nẵng', 'Cần Thơ', 'Vinh']
    selected_city = st.selectbox("Chọn thành phố để xem lịch sử:", cities, index=cities.index(st.session_state.selected_city))
    st.session_state.selected_city = selected_city

    # Historical AQI, Wind Speed, Humidity plot
    city_data = df[df['city'] == selected_city].sort_values('timestamp')
    if not city_data.empty:
        st.markdown(f"<h3 class='text-xl font-semibold text-gray-700'>Lịch sử tại {selected_city}</h3>", unsafe_allow_html=True)
        try:
            fig = px.line(city_data, x='timestamp', y=['aqi', 'wind_speed', 'humidity'],
                          title=f"Lịch sử AQI, Tốc độ gió, Độ ẩm tại {selected_city}",
                          labels={'timestamp': 'Thời gian', 'value': 'Giá trị', 'variable': 'Biến'})
            fig.update_layout(
                xaxis_tickangle=45,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='black'),  # Set all text to black
                title_font=dict(color='black'),
                xaxis_title_font=dict(color='black'),
                yaxis_title_font=dict(color='black'),
                xaxis_tickfont=dict(color='black'),
                yaxis_tickfont=dict(color='black')
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error plotting data: {str(e)}. Ensure 'aqi', 'wind_speed', and 'humidity' are numeric.")
    else:
        st.markdown("<p class='text-gray-600'>Chưa có dữ liệu lịch sử cho thành phố này.</p>", unsafe_allow_html=True)

# Main content
st.markdown("<div class='container mx-auto px-4'>", unsafe_allow_html=True)

# Display current time
current_time = get_vietnam_time().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"<p class='text-lg text-gray-600'>Thời gian hiện tại: {current_time}</p>", unsafe_allow_html=True)

# Prediction section
st.markdown("<h2 class='text-2xl font-semibold text-gray-800 mt-6'>Dự đoán AQI, Tốc độ gió, và Độ ẩm</h2>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])

with col1:
    selected_city_pred = st.selectbox("Chọn thành phố để dự đoán:", cities,
                                      index=cities.index(st.session_state.selected_city_pred))
    st.session_state.selected_city_pred = selected_city_pred

    future_date = st.date_input("Chọn ngày dự đoán:", min_value=datetime.today().date(),
                                value=st.session_state.future_date)
    st.session_state.future_date = future_date

    future_time = st.time_input("Chọn giờ bắt đầu:", value=st.session_state.future_time, step=1800)
    st.session_state.future_time = future_time

    forecast_hours = st.slider("Dự đoán trong bao nhiêu giờ tiếp theo:", 1, 12, 6)

with col2:
    if st.button("Dự đoán"):
        future_datetime = datetime.combine(future_date, future_time)
        predictions = create_sequence_for_prediction(selected_city_pred, future_datetime, df, features, scaler, forecast_hours=forecast_hours)
        if predictions is not None:
            pred_data = []
            for h, pred in enumerate(predictions):
                aqi, wind_speed, humidity = pred
                pred_time = future_datetime + timedelta(hours=h)
                aqi_category, bg_color, health_impact = get_aqi_category(aqi)
                pred_data.append({
                    'Thời gian': pred_time,
                    'AQI': aqi,
                    'Tốc độ gió (km/h)': wind_speed,
                    'Độ ẩm (%)': humidity,
                    'Mức độ AQI': aqi_category
                })

            # Display prediction table
            pred_df = pd.DataFrame(pred_data)
            st.markdown("<h3 class='text-xl font-semibold text-gray-800'>Kết quả dự đoán</h3>", unsafe_allow_html=True)
            st.dataframe(pred_df.style.format({'AQI': '{:.1f}', 'Tốc độ gió (km/h)': '{:.1f}', 'Độ ẩm (%)': '{:.1f}'}))

            # Prediction plot
            fig = px.line(pred_df, x='Thời gian', y=['AQI', 'Tốc độ gió (km/h)', 'Độ ẩm (%)'],
                          title=f"Dự đoán tại {selected_city_pred}",
                          labels={'Thời gian': 'Thời gian', 'value': 'Giá trị', 'variable': 'Biến'})
            fig.update_layout(
                xaxis_tickangle=45,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='black'),  # Set all text to black
                title_font=dict(color='black'),
                xaxis_title_font=dict(color='black'),
                yaxis_title_font=dict(color='black'),
                xaxis_tickfont=dict(color='black'),
                yaxis_tickfont=dict(color='black')
            )
            st.plotly_chart(fig, use_container_width=True)

            # Warning for dangerous AQI
            if any(pred_df['AQI'] > 300):
                st.markdown("<p class='text-red-600 font-bold'>CẢNH BÁO: AQI dự đoán vượt ngưỡng nguy hiểm (>300)!</p>", unsafe_allow_html=True)

            # Export CSV button
            if st.button("Xuất báo cáo CSV"):
                csv_filename = f"predictions_{selected_city_pred}_{future_datetime.strftime('%Y%m%d_%H%M')}.csv"
                pred_df.to_csv(csv_filename, index=False)
                st.success(f"Đã xuất báo cáo vào {csv_filename}!")

# Retrain model button
if st.button("Huấn luyện lại mô hình"):
    with open(RETRAIN_FLAG, 'w', encoding='utf-8') as f:
        f.write("retrain")
    st.info("Đã yêu cầu huấn luyện lại mô hình. Vui lòng chạy lại file `huanluyen6.py` hoặc đợi nếu đã thiết lập tự động.")

st.markdown("</div>", unsafe_allow_html=True)

# Auto-refresh
time.sleep(1)
st.rerun()