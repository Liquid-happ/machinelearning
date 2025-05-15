import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import pickle
import os

# Đường dẫn tuyệt đối đến file CSV
CSV_FILE = os.path.join(os.getcwd(), 'aqicn_aqi_data.csv')

# Kiểm tra xem file có tồn tại không
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"File {CSV_FILE} không tồn tại. Hãy chạy fetch_aqicn_data.py trước.")

try:
    df = pd.read_csv(CSV_FILE, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(CSV_FILE, encoding='latin1')

# Kiểm tra và điền giá trị mặc định cho cột 'city' nếu rỗng
if df['city'].isna().all() or df['city'].empty:
    df['city'] = 'Hà Nội'

# Tiền xử lý dữ liệu
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month

# Chuyển đổi cột số thành float nếu có
df['aqi'] = pd.to_numeric(df['aqi'], errors='coerce')

# Xóa các hàng có giá trị NaN trong cột aqi
df = df.dropna(subset=['aqi'])

# Mã hóa cột 'city'
le = LabelEncoder()
df['city'] = le.fit_transform(df['city'])

# In các nhãn đã học để kiểm tra
print("Các nhãn đã học cho 'city':", le.classes_)

# Chọn đặc trưng và nhãn
features = ['hour', 'day', 'month', 'city']
X = df[features]
y = df['aqi']

# Kiểm tra xem có đủ dữ liệu không
if X.empty or y.empty:
    raise ValueError("Dữ liệu không đủ để huấn luyện mô hình. Vui lòng thu thập thêm dữ liệu.")

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Đánh giá mô hình
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)
print(f'R^2 Score: {score:.4f}')

# Hàm phân loại AQI và đưa ra lời khuyên
def classify_aqi(aqi_value):
    if aqi_value <= 50:
        category = "Tốt"
        advice = "Chất lượng không khí tốt, bạn có thể thoải mái tham gia các hoạt động ngoài trời."
    elif aqi_value <= 100:
        category = "Trung bình"
        advice = "Chất lượng không khí ở mức chấp nhận được, nhưng những người nhạy cảm (trẻ em, người già) nên hạn chế thời gian ở ngoài trời."
    elif aqi_value <= 150:
        category = "Không lành mạnh cho nhóm nhạy cảm"
        advice = "Nhóm nhạy cảm (người già, trẻ em, người mắc bệnh hô hấp) nên giảm hoạt động ngoài trời. Người bình thường vẫn có thể hoạt động nhưng cần chú ý."
    elif aqi_value <= 200:
        category = "Không lành mạnh"
        advice = "Mọi người nên hạn chế hoạt động ngoài trời, đặc biệt là các hoạt động gắng sức. Đeo khẩu trang khi ra ngoài."
    elif aqi_value <= 300:
        category = "Rất không lành mạnh"
        advice = "Tránh mọi hoạt động ngoài trời. Ở trong nhà, đóng cửa sổ và sử dụng máy lọc không khí nếu có."
    else:
        category = "Nguy hiểm"
        advice = "Chất lượng không khí cực kỳ nguy hiểm. Ở trong nhà, sử dụng máy lọc không khí và tránh mọi tiếp xúc với không khí bên ngoài."
    return category, advice

# Phân loại và đưa ra lời khuyên cho các giá trị dự đoán trên tập kiểm tra
print("\nPhân loại chất lượng không khí trên tập kiểm tra:")
for idx, (actual, predicted) in enumerate(zip(y_test, y_pred)):
    category, advice = classify_aqi(predicted)
    print(f"Mẫu {idx+1}: AQI thực tế = {actual:.2f}, AQI dự đoán = {predicted:.2f}")
    print(f"Phân loại: {category}")
    print(f"Lời khuyên: {advice}\n")

# Lưu mô hình để sử dụng trong ứng dụng web
with open('aqi_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Mô hình đã được huấn luyện và lưu thành công!")