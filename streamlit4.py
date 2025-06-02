import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os
import time
import logging
import json

# Cấu hình logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

# Đường dẫn tệp
CSV_FILE = os.path.join(os.getcwd(), 'all_cities_aqi_data.csv')
MODEL_FILE = os.path.join(os.getcwd(), 'models', 'aqi_multioutput_xgboost_model.pkl')
FEATURES_FILE = os.path.join(os.getcwd(), 'models', 'features.txt')
RETRAIN_FLAG = os.path.join(os.getcwd(), 'retrain_flag.txt')

# Đọc danh sách đặc trưng
try:
    with open(FEATURES_FILE, 'r', encoding='utf-8') as f:
        features = f.read().split(',')
except FileNotFoundError:
    st.error("Không tìm thấy tệp đặc trưng. Vui lòng huấn luyện mô hình trước.")
    logging.error("Không tìm thấy tệp đặc trưng.")
    st.stop()

# Tải mô hình
try:
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    st.error("Không tìm thấy mô hình. Vui lòng huấn luyện mô hình trước.")
    logging.error("Không tìm thấy mô hình.")
    st.stop()

# Đọc dữ liệu lịch sử
try:
    df = pd.read_csv(CSV_FILE, encoding='utf-8')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['aqi'] = df['aqi'].astype(float)  # Đảm bảo AQI là số
except FileNotFoundError:
    st.error("Không tìm thấy tệp dữ liệu. Vui lòng chạy script thu thập dữ liệu.")
    logging.error("Không tìm thấy tệp dữ liệu.")
    st.stop()
except Exception as e:
    st.error(f"Lỗi khi đọc dữ liệu: {str(e)}")
    logging.error(f"Lỗi khi đọc dữ liệu: {str(e)}")
    st.stop()

# Lấy thời gian Việt Nam
def get_vietnam_time():
    return datetime.now(ZoneInfo("Asia/Bangkok"))

# Ánh xạ màu
TAILWIND_TO_PLOTLY_COLORS = {
    'green-600': '#16a34a',
    'yellow-400': '#facc15',
    'orange-600': '#ea580c',
    'red-500': '#ef4444',
    'purple-600': '#9333ea',
    'red-800': '#991b1b'
}

COLOR_MAP = {
    'Hà Nội': '#1f77b4',
    'Hồ Chí Minh': '#ff7f0e',
    'Đà Nẵng': '#2ca02c',
    'Cần Thơ': '#d62728',
    'Vinh': '#9467bd'
}

# Phân loại AQI
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Tốt", "bg-green-600", "Không khí sạch, không gây nguy hiểm."
    elif aqi <= 100:
        return "Trung bình", "bg-yellow-400", "Chất lượng không khí chấp nhận được, nhưng một số chất ô nhiễm có thể ảnh hưởng đến nhóm nhạy cảm."
    elif aqi <= 150:
        return "Không tốt cho nhóm nhạy cảm", "bg-orange-600", "Nhóm nhạy cảm có thể gặp vấn đề sức khỏe."
    elif aqi <= 200:
        return "Có hại", "bg-red-500", "Mọi người bắt đầu cảm nhận được ảnh hưởng sức khỏe."
    elif aqi <= 300:
        return "Rất có hại", "bg-purple-600", "Cảnh báo sức khỏe: mọi người bị ảnh hưởng nghiêm trọng hơn."
    else:
        return "Nguy hiểm", "bg-red-800", "Cảnh báo sức khỏe khẩn cấp: toàn bộ dân số có nguy cơ bị ảnh hưởng."

# Khởi tạo session state
if 'selected_city' not in st.session_state:
    st.session_state.selected_city = 'Hà Nội'
if 'selected_city_pred' not in st.session_state:
    st.session_state.selected_city_pred = 'Hà Nội'
if 'future_date' not in st.session_state:
    st.session_state.future_date = datetime.today().date()
if 'future_time' not in st.session_state:
    current_time = get_vietnam_time()
    rounded_minutes = (current_time.minute // 15) * 15
    st.session_state.future_time = current_time.replace(minute=rounded_minutes, second=0, microsecond=0).time()

# Cấu hình ứng dụng Streamlit
st.set_page_config(page_title="Dự đoán AQI Việt Nam", layout="wide")
st.markdown("""
    <style>
        body { font-family: Arial, sans-serif; }
        .aqi-gauge { text-align: center; font-size: 2rem; font-weight: bold; padding: 0.5rem; border-radius: 0.5rem; color: white; }
        .sidebar .sidebar-content { background-color: #f0f4f8; }
        .prediction-box { background-color: #f0f4f8; padding: 1rem; border-radius: 0.5rem; }
        .container { max-width: 100%; padding-left: 0.5rem; padding-right: 0.5rem; }
        canvas { max-width: 100% !important; height: auto !important; }
        @media (max-width: 640px) {
            h1 { font-size: 1.5rem; }
            h2 { font-size: 1.25rem; }
            h3 { font-size: 1rem; }
            p, label { font-size: 0.875rem; }
            .aqi-gauge { font-size: 1.5rem; }
            select, input, button { font-size: 0.875rem; padding: 0.5rem; }
            .stButton>button { width: 100%; }
        }
    </style>
""", unsafe_allow_html=True)

# Tiêu đề chính
st.markdown("""
    <div class="text-center py-4">
        <h1 class="text-3xl font-bold text-gray-800">Dự đoán Chỉ số Chất lượng Không khí (AQI)</h1>
        <p class="text-base text-gray-600">Theo dõi và dự đoán chất lượng không khí tại Việt Nam</p>
    </div>
""", unsafe_allow_html=True)

# Thanh sidebar
with st.sidebar:
    st.markdown("<h2 class='text-xl font-semibold text-gray-800'>Lịch sử AQI</h2>", unsafe_allow_html=True)
    cities = ['Hà Nội', 'Hồ Chí Minh', 'Đà Nẵng', 'Cần Thơ', 'Vinh']
    selected_city = st.selectbox("Chọn thành phố để xem lịch sử:", cities, index=cities.index(st.session_state.selected_city))
    st.session_state.selected_city = selected_city

    # Biểu đồ lịch sử AQI
    city_data = df[df['city'] == selected_city].sort_values('timestamp')
    if not city_data.empty:
        st.markdown(f"<h3 class='text-lg font-semibold text-gray-700'>Lịch sử AQI tại {selected_city}</h3>", unsafe_allow_html=True)
        fig = px.line(city_data, x='timestamp', y='aqi', title=f"AQI tại {selected_city}",
                      labels={'timestamp': 'Thời gian', 'aqi': 'AQI'},
                      color_discrete_sequence=['#1f77b4'])
        fig.update_layout(
            xaxis_tickangle=45,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12, family="Arial"),
            xaxis=dict(showgrid=True, gridcolor='lightgray', title_font_size=12),
            yaxis=dict(showgrid=True, gridcolor='lightgray', title="AQI", title_font_size=12),
            hovermode='x unified',
            showlegend=True,
            margin=dict(l=20, r=20, t=60, b=40),
            height=300
        )
        fig.update_traces(
            hovertemplate="Thời gian: %{x}<br>AQI: %{y:.1f}<extra></extra>",
            line=dict(width=2)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("<p class='text-gray-600'>Chưa có dữ liệu lịch sử cho thành phố này.</p>", unsafe_allow_html=True)

# Nội dung chính
st.markdown("<div class='container mx-auto px-2'>", unsafe_allow_html=True)

# Thời gian hiện tại
current_time = get_vietnam_time().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"<p class='text-base text-gray-600'>Thời gian hiện tại: {current_time}</p>", unsafe_allow_html=True)

# Biểu đồ so sánh AQI (ChartJS)
st.markdown("<h3 class='text-lg font-semibold text-gray-700'>So sánh AQI giữa các thành phố</h3>", unsafe_allow_html=True)
@st.cache_data
def prepare_chart_data(df, cities):
    chart_data = {"labels": df['timestamp'].dt.strftime('%Y-%m-%d').unique().tolist()[-30:], "datasets": []}
    for city in cities:
        city_data = df[df['city'] == city][['timestamp', 'aqi']].sort_values('timestamp')
        city_data = city_data.groupby(city_data['timestamp'].dt.strftime('%Y-%m-%d'))['aqi'].mean().reset_index()
        city_data = city_data.tail(30)  # Giới hạn 30 ngày
        chart_data['datasets'].append({
            "label": city,
            "data": city_data['aqi'].tolist(),
            "borderColor": COLOR_MAP[city],
            "backgroundColor": COLOR_MAP[city],
            "fill": False,
            "lineWidth": 2
        })
    return chart_data

chart_data = prepare_chart_data(df, cities)
chart_js_code = f"""
    <canvas id="aqiComparisonChart" style="max-height: 300px;"></canvas>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
    <script>
        const ctx = document.getElementById('aqiComparisonChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {json.dumps(chart_data, ensure_ascii=True)},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'So sánh AQI',
                        font: {{ size: 14, family: 'Arial' }}
                    }},
                    legend: {{
                        display: true,
                        position: 'top',
                        labels: {{ font: {{ size: 12 }} }}
                    }},
                    tooltip: {{
                        enabled: true,
                        callbacks: {{
                            label: function(context) {{
                                return context.dataset.label + ': ' + context.parsed.y.toFixed(1) + ' AQI';
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'Thời gian',
                            font: {{ size: 12 }}
                        }},
                        grid: {{ display: true, color: 'lightgray' }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'AQI',
                            font: {{ size: 12 }}
                        }},
                        grid: {{ display: true, color: 'lightgray' }},
                        beginAtZero: true
                    }}
                }}
            }}
        }});
    </script>
"""
st.markdown(chart_js_code, unsafe_allow_html=True)

# Phần dự đoán
st.markdown("<h2 class='text-xl font-semibold text-gray-800 mt-4'>Dự đoán AQI, Tốc độ gió, và Độ ẩm</h2>", unsafe_allow_html=True)
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
            'day_of_week': future_datetime.weekday(),
            'is_rainy_season': 1 if future_datetime.month in [5, 6, 7, 8, 9, 10] else 0
        }
        for city_col in [col for col in features if col.startswith('city_')]:
            city_name = city_col.split('city_')[1]
            input_data[city_col] = 1 if city_name == selected_city_pred else 0

        try:
            input_df = pd.DataFrame([input_data], columns=features)
            predictions = model.predict(input_df)[0]
            aqi, wind_speed, humidity = predictions
            aqi_category, bg_color, health_impact = get_aqi_category(aqi)
            plotly_color = TAILWIND_TO_PLOTLY_COLORS[bg_color.replace('bg-', '')]

            st.markdown(f"""
                <div class='prediction-box'>
                    <h3 class='text-lg font-semibold text-gray-800'>Dự đoán tại {selected_city_pred}</h3>
                    <p class='text-base text-gray-600'>Thời gian: {future_datetime.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <div class='aqi-gauge {bg_color}'>{aqi:.1f}</div>
                    <p class='text-base font-semibold text-gray-700'>Mức độ AQI: {aqi_category}</p>
                    <p class='text-gray-600'>{health_impact}</p>
                    <p class='text-base text-gray-600'>Tốc độ gió: {wind_speed:.1f} km/h</p>
                    <p class='text-base text-gray-600'>Độ ẩm: {humidity:.1f} %</p>
                </div>
            """, unsafe_allow_html=True)

            # Ghi log dự đoán
            logging.info(f"Dự đoán cho {selected_city_pred} tại {future_datetime}: AQI={aqi:.1f}, Tốc độ gió={wind_speed:.1f}, Độ ẩm={humidity:.1f}")

            # Biểu đồ gauge AQI
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=aqi,
                title={'text': "Chỉ số AQI", 'font': {'size: 16, 'family': "Arial"}},
                number={'font': {'size: 32, 'color': plotly_color}},
                gauge={
                    'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': "black", 'tickfont': {'size: 12}},
                    'bar': {'color': plotly_color, 'thickness': 0.2},
                    'steps': [
                        {'range': [0, 50], 'color': "#16a34a"},
                        {'range': [50, 100], 'color': "#facc15"},
                        {'range': [100, 150], 'color': "#ea580c"},
                        {'range': [150, 200], 'color': "#ef4444"},
                        {'range': [200, 300], 'color': "#9333ea"},
                        {'range': [300, 500], 'color': "#991b1b"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': aqi
                    }
                }
            ))
            fig_gauge.update_layout(
                margin=dict(l=20, r=20, t=50, b=20),
                height=250,
                font=dict(family="Arial", size=12),
                paper_bgcolor='white'
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {str(e)}")
            logging.error(f"Lỗi dự đoán: {str(e)}")

# Nút huấn luyện lại mô hình
if st.button("Huấn luyện lại mô hình"):
    with open(RETRAIN_FLAG, 'w', encoding='utf-8') as f:
        f.write("retrain")
    st.info("Đã yêu cầu huấn luyện lại mô hình. Vui lòng chạy lại file `huanluyen4.py` hoặc đợi nếu đã thiết lập tự động.")
    logging.info("Yêu cầu huấn luyện lại mô hình.")

# Tự động làm mới mỗi 5 phút
if time.time() % 300 < 1:
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
