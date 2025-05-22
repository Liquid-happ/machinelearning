import streamlit4 as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os
import time

# Đường dẫn đến file dữ liệu, mô hình và đặc trưng
CSV_FILE = os.path.join(os.getcwd(), 'all_cities_aqi_data.csv')
MODEL_FILE = os.path.join(os.getcwd(), 'models', 'aqi_multioutput_xgboost_model.pkl')
FEATURES_FILE = os.path.join(os.getcwd(), 'models', 'features.txt')
RETRAIN_FLAG = os.path.join(os.getcwd(), 'retrain_flag.txt')

# Đọc danh sách đặc trưng
try:
    with open(FEATURES_FILE, 'r', encoding='utf-8') as f:
        features = f.read().split(',')
except FileNotFoundError:
    st.error("File đặc trưng không tồn tại. Vui lòng huấn luyện mô hình trước.")
    st.stop()

# Tải mô hình
try:
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    st.error("Mô hình không tồn tại. Vui lòng huấn luyện mô hình trước.")
    st.stop()

# Đọc dữ liệu lịch sử
try:
    df = pd.read_csv(CSV_FILE, encoding='utf-8')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
except FileNotFoundError:
    st.error("File dữ liệu không tồn tại. Vui lòng chạy script thu thập dữ liệu trước.")
    st.stop()

# Hàm lấy thời gian Việt Nam
def get_vietnam_time():
    return datetime.now(ZoneInfo("Asia/Bangkok"))

# Ánh xạ màu Tailwind sang màu Plotly
TAILWIND_TO_PLOTLY_COLORS = {
    'green-500': '#22c55e',
    'yellow-500': '#eab308',
    'orange-500': '#f97316',
    'red-500': '#ef4444',
    'purple-500': '#a855f7',
    'maroon-500': '#7f1d1d'
}

# Hàm xác định mức độ AQI
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

# Khởi tạo session state
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

# Giao diện Streamlit với Tailwind CSS
st.set_page_config(page_title="Dự đoán AQI Việt Nam", layout="wide")
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        body { font-family: Arial, sans-serif; }
        .aqi-gauge { text-align: center; font-size: 3rem; font-weight: bold; padding: 1rem; border-radius: 0.5rem; color: white; }
        .sidebar .sidebar-content { background-color: #f8f9fa; }
        .prediction-box { background-color: #f0f4f8; padding: 1.5rem; border-radius: 0.5rem; }
    </style>
""", unsafe_allow_html=True)

# Tiêu đề chính
st.markdown("""
    <div class="text-center py-6">
        <h1 class="text-4xl font-bold text-gray-800">Dự đoán Chỉ số Chất lượng Không khí (AQI)</h1>
        <p class="text-lg text-gray-600">Theo dõi và dự đoán chất lượng không khí, tốc độ gió, và độ ẩm tại các thành phố lớn ở Việt Nam</p>
    </div>
""", unsafe_allow_html=True)

# Thanh sidebar
with st.sidebar:
    st.markdown("<h2 class='text-2xl font-semibold text-gray-800'>Lịch sử AQI</h2>", unsafe_allow_html=True)
    cities = ['Hà Nội', 'Hồ Chí Minh', 'Đà Nẵng', 'Cần Thơ', 'Vinh']
    selected_city = st.selectbox("Chọn thành phố để xem lịch sử:", cities, index=cities.index(st.session_state.selected_city))
    st.session_state.selected_city = selected_city

    # Biểu đồ lịch sử AQI
    city_data = df[df['city'] == selected_city].sort_values('timestamp')
    if not city_data.empty:
        st.markdown(f"<h3 class='text-xl font-semibold text-gray-700'>Lịch sử AQI tại {selected_city}</h3>", unsafe_allow_html=True)
        fig = px.line(city_data, x='timestamp', y='aqi', title=f"AQI tại {selected_city}",
                      labels={'timestamp': 'Thời gian', 'aqi': 'AQI'})
        fig.update_layout(xaxis_tickangle=45, plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("<p class='text-gray-600'>Chưa có dữ liệu lịch sử cho thành phố này.</p>", unsafe_allow_html=True)

# Nội dung chính
st.markdown("<div class='container mx-auto px-4'>", unsafe_allow_html=True)

# Hiển thị thời gian hiện tại
current_time = get_vietnam_time().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"<p class='text-lg text-gray-600'>Thời gian hiện tại: {current_time}</p>", unsafe_allow_html=True)

# Phần dự đoán
st.markdown("<h2 class='text-2xl font-semibold text-gray-800 mt-6'>Dự đoán AQI, Tốc độ gió, và Độ ẩm</h2>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])

with col1:
    selected_city_pred = st.selectbox("Chọn thành phố để dự đoán:", cities,
                                      index=cities.index(st.session_state.selected_city_pred))
    st.session_state.selected_city_pred = selected_city_pred

    future_date = st.date_input("Chọn ngày dự đoán:", min_value=datetime.today(),
                                value=st.session_state.future_date)
    st.session_state.future_date = future_date

    future_time = st.time_input("Chọn giờ dự đoán:", value=st.session_state.future_time, step=1800)
    st.session_state.future_time = future_time

with col2:
    if st.button("Dự đoán"):
        future_datetime = datetime.combine(future_date, future_time)
        input_data = {
            'year': future_datetime.year,
            'month': future_datetime.month,
            'day': future_datetime.day,
            'hour': future_datetime.hour,
            'day_of_week': future_datetime.weekday()
        }
        for city_col in [col for col in features if col.startswith('city_')]:
            city_name = city_col.split('city_')[1]
            input_data[city_col] = 1 if city_name == selected_city_pred else 0

        input_df = pd.DataFrame([input_data], columns=features)
        predictions = model.predict(input_df)[0]
        aqi, wind_speed, humidity = predictions
        aqi_category, bg_color, health_impact = get_aqi_category(aqi)
        plotly_color = TAILWIND_TO_PLOTLY_COLORS[bg_color.replace('bg-', '')]

        st.markdown(f"""
            <div class='prediction-box'>
                <h3 class='text-xl font-semibold text-gray-800'>Dự đoán tại {selected_city_pred}</h3>
                <p class='text-lg text-gray-600'>Thời gian: {future_datetime.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <div class='aqi-gauge {bg_color}'>{aqi:.1f}</div>
                <p class='text-lg font-semibold text-gray-700'>Mức độ AQI: {aqi_category}</p>
                <p class='text-gray-600'>{health_impact}</p>
                <p class='text-lg text-gray-600'>Tốc độ gió: {wind_speed:.1f} km/h</p>
                <p class='text-lg text-gray-600'>Độ ẩm: {humidity:.1f} %</p>
            </div>
        """, unsafe_allow_html=True)

        # Biểu đồ gauge AQI
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=aqi,
            title={'text': "Chỉ số AQI"},
            gauge={
                'axis': {'range': [0, 500]},
                'bar': {'color': plotly_color},
                'steps': [
                    {'range': [0, 50], 'color': "#22c55e"},
                    {'range': [50, 100], 'color': "#eab308"},
                    {'range': [100, 150], 'color': "#f97316"},
                    {'range': [150, 200], 'color': "#ef4444"},
                    {'range': [200, 300], 'color': "#a855f7"},
                    {'range': [300, 500], 'color': "#7f1d1d"}
                ]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

# Nút huấn luyện lại mô hình
if st.button("Huấn luyện lại mô hình"):
    with open(RETRAIN_FLAG, 'w', encoding='utf-8') as f:
        f.write("retrain")
    st.info("Đã yêu cầu huấn luyện lại mô hình. Vui lòng chạy lại file `huanluyen3.py` hoặc đợi nếu đã thiết lập tự động.")

st.markdown("</div>", unsafe_allow_html=True)

# Tự động làm mới
time.sleep(1)
st.rerun()