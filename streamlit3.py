import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os
import time

# Đường dẫn đến file dữ liệu, mô hình và đặc trưng
CSV_FILE = os.path.join(os.getcwd(), 'all_cities_aqi_data.csv')
MODEL_FILE = os.path.join(os.getcwd(), 'models', 'aqi_xgboost_model.pkl')
FEATURES_FILE = os.path.join(os.getcwd(), 'models', 'features.txt')
RETRAIN_FLAG = os.path.join(os.getcwd(), 'retrain_flag.txt')

# Đọc danh sách đặc trưng
with open(FEATURES_FILE, 'r', encoding='utf-8') as f:
    features = f.read().split(',')

# Tải mô hình
model = joblib.load(MODEL_FILE)

# Đọc dữ liệu lịch sử với mã hóa utf-8
df = pd.read_csv(CSV_FILE, encoding='utf-8')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Tính giá trị trung bình của wind_speed và humidity
df['wind_speed'] = df['wind_speed'].str.replace(' km/h', '', regex=False).astype(float)
df['humidity'] = df['humidity'].str.replace('%', '', regex=False).astype(float)
avg_wind_speed = df['wind_speed'].mean()
avg_humidity = df['humidity'].mean()

# Kiểm tra và xử lý dữ liệu trống
if df.empty or pd.isna(avg_wind_speed) or pd.isna(avg_humidity):
    st.error("Không có đủ dữ liệu lịch sử để tính trung bình. Sử dụng giá trị mặc định.")
    wind_speed, humidity = 5.0, 70.0
else:
    wind_speed = avg_wind_speed
    humidity = avg_humidity

# Hàm lấy thời gian hiện tại
def get_vietnam_time():
    return datetime.now(ZoneInfo("Asia/Bangkok"))

# Khởi tạo session state để lưu trạng thái lựa chọn
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

# Giao diện Streamlit
st.title("Dự đoán Chỉ số Chất lượng Không khí (AQI)")

# Hiển thị thời gian hiện tại
current_time = get_vietnam_time().strftime("%Y-%m-%d %H:%M:%S")
st.write(f"Thời gian hiện tại: {current_time}")

# Sidebar: Lựa chọn thành phố và hiển thị lịch sử
st.sidebar.header("Lịch sử AQI")
cities = ['Hà Nội', 'Hồ Chí Minh', 'Đà Nẵng', 'Cần Thơ', 'Vinh']
selected_city = st.sidebar.selectbox("Chọn thành phố để xem lịch sử:", cities, index=cities.index(st.session_state.selected_city))
st.session_state.selected_city = selected_city

# Hiển thị biểu đồ lịch sử AQI
city_data = df[df['city'] == selected_city].sort_values('timestamp')
if not city_data.empty:
    st.sidebar.subheader(f"Lịch sử AQI tại {selected_city}")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=city_data, x='timestamp', y='aqi', ax=ax, color='blue')
    ax.set_title(f"AQI tại {selected_city}")
    ax.set_xlabel("Thời gian")
    ax.set_ylabel("AQI")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.sidebar.pyplot(fig)
else:
    st.sidebar.write("Chưa có dữ liệu lịch sử cho thành phố này.")

# Dự đoán AQI
st.header("Dự đoán AQI")
selected_city_pred = st.selectbox("Chọn thành phố để dự đoán:", cities, index=cities.index(st.session_state.selected_city_pred))
st.session_state.selected_city_pred = selected_city_pred

# Chọn ngày và giờ dự đoán
future_date = st.date_input("Chọn ngày dự đoán:", min_value=datetime.today(), value=st.session_state.future_date)
st.session_state.future_date = future_date

# Chọn giờ dự đoán với khoảng cách 30 phút
future_time = st.time_input(
    "Chọn giờ dự đoán:",
    value=st.session_state.future_time,
    step=1800  # 30 phút = 1800 giây
)
st.session_state.future_time = future_time
future_datetime = datetime.combine(future_date, future_time)

# Dự đoán khi nhấn nút
if st.button("Dự đoán AQI"):
    input_data = {
        'year': future_datetime.year,
        'month': future_datetime.month,
        'day': future_datetime.day,
        'hour': future_datetime.hour,
        'day_of_week': future_datetime.weekday(),
        'wind_speed': wind_speed,
        'humidity': humidity
    }

    for city_col in [col for col in features if col.startswith('city_')]:
        city_name = city_col.split('city_')[1]
        input_data[city_col] = 1 if city_name == selected_city_pred else 0

    input_df = pd.DataFrame([input_data], columns=features)
    prediction = model.predict(input_df)[0]
    st.success(f"Dự đoán AQI tại {selected_city_pred} vào {future_datetime.strftime('%Y-%m-%d %H:%M:%S')}: **{prediction:.2f}**")

    # Đánh giá mức độ AQI
    if prediction <= 50:
        st.write("**Chất lượng không khí: Tốt**")
    elif prediction <= 100:
        st.write("**Chất lượng không khí: Trung bình**")
    elif prediction <= 150:
        st.write("**Chất lượng không khí: Không tốt cho nhóm nhạy cảm**")
    elif prediction <= 200:
        st.write("**Chất lượng không khí: Có hại**")
    elif prediction <= 300:
        st.write("**Chất lượng không khí: Rất có hại**")
    else:
        st.write("**Chất lượng không khí: Nguy hiểm**")

# Nút huấn luyện lại mô hình
if st.button("Huấn luyện lại mô hình"):
    with open(RETRAIN_FLAG, 'w', encoding='utf-8') as f:
        f.write("retrain")
    st.info("Đã yêu cầu huấn luyện lại mô hình. Vui lòng chạy lại file `train_aqi_model.py` hoặc đợi nếu bạn đã thiết lập tự động huấn luyện.")

# Tự động làm mới để cập nhật thời gian
time.sleep(1)
st.rerun()
