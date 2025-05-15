import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import os
import plotly.express as px

# Tải mô hình và LabelEncoder
with open('aqi_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Đường dẫn đến file CSV
CSV_FILE = os.path.join(os.getcwd(), 'aqicn_aqi_data.csv')

# Hàm cào dữ liệu thời gian thực AQI từ aqicn.org
def fetch_latest_data():
    url = 'https://aqicn.org/city/vietnam/hanoi/'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        aqi_div = soup.find('div', {'id': 'aqiwgtvalue'})
        aqi = float(aqi_div.text) if aqi_div else 0

        return {
            'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            'aqi': aqi,
            'city': 'Hà Nội'
        }
    except Exception as e:
        st.error(f"Lỗi khi lấy dữ liệu từ AQICN.org: {e}")
        return None

# Hàm lấy dữ liệu gần nhất từ file CSV
def fetch_latest_from_csv():
    if not os.path.exists(CSV_FILE):
        st.warning("File aqicn_aqi_data.csv không tồn tại. Hãy chạy fetch_aqicn_data.py trước.")
        return None
    try:
        df = pd.read_csv(CSV_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        latest_row = df.iloc[-1]
        return {
            'timestamp': latest_row['timestamp'].strftime('%Y-%m-%dT%H:%M:%S'),
            'aqi': float(latest_row['aqi']),
            'city': latest_row['city']
        }
    except Exception as e:
        st.error(f"Lỗi khi đọc dữ liệu từ CSV: {e}")
        return None

# Hàm lấy dữ liệu 24 giờ qua để vẽ biểu đồ
def fetch_last_24h_data():
    if not os.path.exists(CSV_FILE):
        return None
    try:
        df = pd.read_csv(CSV_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        cutoff_time = pd.to_datetime('now') - pd.Timedelta(hours=24)
        df = df[df['timestamp'] >= cutoff_time]
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc dữ liệu 24 giờ từ CSV: {e}")
        return None

# Hàm phân loại AQI và trả về màu sắc
def classify_aqi(aqi_value):
    if aqi_value <= 50:
        category = "Tốt"
        advice = "Chất lượng không khí tốt, bạn có thể thoải mái tham gia các hoạt động ngoài trời."
        color = "#00FF00"  # Xanh lá
    elif aqi_value <= 100:
        category = "Trung bình"
        advice = "Chất lượng không khí ở mức chấp nhận được, nhưng những người nhạy cảm (trẻ em, người già) nên hạn chế thời gian ở ngoài trời."
        color = "#FFFF00"  # Vàng
    elif aqi_value <= 150:
        category = "Không lành mạnh cho nhóm nhạy cảm"
        advice = "Nhóm nhạy cảm (người già, trẻ em, người mắc bệnh hô hấp) nên giảm hoạt động ngoài trời. Người bình thường vẫn có thể hoạt động nhưng cần chú ý."
        color = "#FFA500"  # Cam
    elif aqi_value <= 200:
        category = "Không lành mạnh"
        advice = "Mọi người nên hạn chế hoạt động ngoài trời, đặc biệt là các hoạt động gắng sức. Đeo khẩu trang khi ra ngoài."
        color = "#FF0000"  # Đỏ
    elif aqi_value <= 300:
        category = "Rất không lành mạnh"
        advice = "Tránh mọi hoạt động ngoài trời. Ở trong nhà, đóng cửa sổ và sử dụng máy lọc không khí nếu có."
        color = "#800080"  # Tím
    else:
        category = "Nguy hiểm"
        advice = "Chất lượng không khí cực kỳ nguy hiểm. Ở trong nhà, sử dụng máy lọc không khí và tránh mọi tiếp xúc với không khí bên ngoài."
        color = "#000000"  # Đen
    return category, advice, color

# Hàm dự đoán AQI
def predict_aqi(hour, day, month, city):
    input_data = pd.DataFrame({
        'hour': [hour],
        'day': [day],
        'month': [month],
        'city': [city]
    })
    try:
        input_data['city'] = le.transform(input_data['city'])
    except ValueError as e:
        st.warning(f"Nhãn 'city' ({city}) chưa được học. Sử dụng giá trị mặc định (0).")
        input_data['city'] = 0
    prediction = model.predict(input_data)[0]
    category, advice, color = classify_aqi(prediction)
    return prediction, category, advice, color

# Tiêu đề ứng dụng
st.markdown("<h1 style='text-align: center; color: #1E90FF;'>Dự đoán Chỉ số Chất lượng Không khí (AQI)</h1>", unsafe_allow_html=True)

# Lấy thời gian hiện tại động
current_time = datetime.now()
weekday = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"][current_time.weekday()]
st.markdown(
    f"<h4 style='text-align: center; color: #555555;'>Thời gian hiện tại: {current_time.strftime('%Y-%m-%d %H:%M:%S')} ({weekday}, {current_time.strftime('%d/%m/%Y')})</h4>",
    unsafe_allow_html=True
)

# Phần hiển thị dữ liệu AQI
st.subheader("Dữ liệu AQI Hiện tại")

col1, col2 = st.columns(2)

# Dữ liệu từ CSV
with col1:
    latest_csv_data = fetch_latest_from_csv()
    if latest_csv_data:
        category_csv, advice_csv, color_csv = classify_aqi(latest_csv_data['aqi'])
        st.markdown(f"**Dữ liệu Gần nhất (CSV - Cập nhật mỗi phút)**")
        st.markdown(f"**Thời gian:** {latest_csv_data['timestamp']}")
        st.markdown(f"**AQI:** <span style='color: {color_csv}; font-weight: bold;'>{latest_csv_data['aqi']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Phân loại:** {category_csv}")

# Dữ liệu thời gian thực
with col2:
    latest_data = fetch_latest_data()
    if latest_data:
        category_rt, advice_rt, color_rt = classify_aqi(latest_data['aqi'])
        st.markdown(f"**Dữ liệu Thời gian thực (AQICN.org)**")
        st.markdown(f"**Thời gian:** {latest_data['timestamp']}")
        st.markdown(f"**AQI:** <span style='color: {color_rt}; font-weight: bold;'>{latest_data['aqi']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Phân loại:** {category_rt}")

# Biểu đồ xu hướng AQI
st.subheader("Xu hướng AQI trong 24 giờ qua")
last_24h_data = fetch_last_24h_data()
if last_24h_data is not None and not last_24h_data.empty:
    fig = px.line(last_24h_data, x='timestamp', y='aqi', title='AQI trong 24 giờ qua',
                  labels={'timestamp': 'Thời gian', 'aqi': 'Chỉ số AQI'})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Không có dữ liệu để vẽ biểu đồ. Hãy chạy fetch_aqicn_data.py để thu thập dữ liệu.")

# Phần nhập liệu và dự đoán
st.subheader("Dự đoán AQI")
with st.form(key='prediction_form'):
    hour = st.number_input("Giờ (0-23)", min_value=0, max_value=23, value=current_time.hour)
    day = st.number_input("Ngày (1-31)", min_value=1, max_value=31, value=current_time.day)
    month = st.number_input("Tháng (1-12)", min_value=1, max_value=12, value=current_time.month)
    city = st.text_input("Thành phố (ví dụ: Hà Nội)", value="Hà Nội")
    submit_button = st.form_submit_button(label="Dự đoán AQI")

# Hiển thị kết quả dự đoán
if submit_button:
    st.markdown("### Kết quả Dự đoán")

    # Dự đoán hiện tại
    prediction, category, advice, color = predict_aqi(hour, day, month, city)
    st.markdown(f"**Dự đoán tại thời gian bạn nhập ({hour}:00 {day}/{month}/2025):**")
    st.markdown(f"<div style='background-color: {color}; color: white; padding: 10px; border-radius: 5px;'>"
                f"AQI: {prediction:.2f}<br>"
                f"Phân loại: {category}<br>"
                f"Lời khuyên: {advice}</div>", unsafe_allow_html=True)

    # Dự đoán sau 1 giờ
    future_time_1h = current_time + timedelta(hours=1)
    future_hour_1h = future_time_1h.hour
    future_day_1h = future_time_1h.day
    future_month_1h = future_time_1h.month
    prediction_1h, category_1h, advice_1h, color_1h = predict_aqi(future_hour_1h, future_day_1h, future_month_1h, city)
    st.markdown(f"**Dự đoán sau 1 giờ ({future_time_1h.strftime('%Y-%m-%d %H:%M:%S')}):**")
    st.markdown(f"<div style='background-color: {color_1h}; color: white; padding: 10px; border-radius: 5px;'>"
                f"AQI: {prediction_1h:.2f}<br>"
                f"Phân loại: {category_1h}<br>"
                f"Lời khuyên: {advice_1h}</div>", unsafe_allow_html=True)

    # Dự đoán sau 2 giờ
    future_time_2h = current_time + timedelta(hours=2)
    future_hour_2h = future_time_2h.hour
    future_day_2h = future_time_2h.day
    future_month_2h = future_time_2h.month
    prediction_2h, category_2h, advice_2h, color_2h = predict_aqi(future_hour_2h, future_day_2h, future_month_2h, city)
    st.markdown(f"**Dự đoán sau 2 giờ ({future_time_2h.strftime('%Y-%m-%d %H:%M:%S')}):**")
    st.markdown(f"<div style='background-color: {color_2h}; color: white; padding: 10px; border-radius: 5px;'>"
                f"AQI: {prediction_2h:.2f}<br>"
                f"Phân loại: {category_2h}<br>"
                f"Lời khuyên: {advice_2h}</div>", unsafe_allow_html=True)

    # Dự đoán ngày mai cùng giờ
    tomorrow = current_time + timedelta(days=1)
    tomorrow_hour = hour
    tomorrow_day = tomorrow.day
    tomorrow_month = tomorrow.month
    prediction_tomorrow, category_tomorrow, advice_tomorrow, color_tomorrow = predict_aqi(tomorrow_hour, tomorrow_day, tomorrow_month, city)
    st.markdown(f"**Dự đoán ngày mai cùng giờ ({tomorrow.strftime('%Y-%m-%d')} {tomorrow_hour}:00):**")
    st.markdown(f"<div style='background-color: {color_tomorrow}; color: white; padding: 10px; border-radius: 5px;'>"
                f"AQI: {prediction_tomorrow:.2f}<br>"
                f"Phân loại: {category_tomorrow}<br>"
                f"Lời khuyên: {advice_tomorrow}</div>", unsafe_allow_html=True)
