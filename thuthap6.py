from playwright.sync_api import sync_playwright
import pandas as pd
import time
import os
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional
import re
import logging

logging.basicConfig(
    filename='crawler.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

CSV_FILE = os.path.join(os.getcwd(), 'all_cities_aqi_data.csv')

CITIES = [
    {"name": "hanoi", "display_name": "Hà Nội", "url": "https://www.iqair.com/vi/vietnam/hanoi/hanoi"},
    {"name": "ho-chi-minh-city", "display_name": "Hồ Chí Minh",
     "url": "https://www.iqair.com/vi/vietnam/ho-chi-minh-city/ho-chi-minh-city"},
    {"name": "da-nang", "display_name": "Đà Nẵng", "url": "https://www.iqair.com/vi/vietnam/da-nang/da-nang"},
    {"name": "can-tho", "display_name": "Cần Thơ", "url": "https://www.iqair.com/vi/vietnam/thanh-pho-can-tho/can-tho"},
    {"name": "vinh", "display_name": "Vinh", "url": "https://www.iqair.com/vi/vietnam/tinh-nghe-an/vinh"}
]

def get_vietnam_time():
    return datetime.now(ZoneInfo("Asia/Bangkok"))

def validate_aqi(aqi: str) -> Optional[str]:
    try:
        aqi_value = int(re.sub(r'\D', '', aqi))
        if 0 <= aqi_value <= 500:
            return str(aqi_value)
    except (ValueError, TypeError):
        pass
    return None

def validate_wind_speed(speed: str) -> Optional[str]:
    try:
        if re.match(r'^\d+(\.\d+)?\s*(km/h|mph)$', speed.strip()):
            speed = speed.strip()
            if 'mph' in speed:
                value = float(re.match(r'^\d+(\.\d+)?', speed).group())
                km_value = value * 1.60934
                return f"{km_value:.1f} km/h"
            return speed
    except (ValueError, TypeError, AttributeError):
        pass
    return None

def validate_humidity(humidity: str) -> Optional[str]:
    try:
        if re.match(r'^\d{1,3}%$', humidity.strip()):
            return humidity.strip()
    except (ValueError, TypeError, AttributeError):
        pass
    return None

def crawl_city_data(page, city: Dict, retries: int = 1) -> Optional[Dict]:
    for attempt in range(retries):
        try:
            logging.info(f"Accessing {city['display_name']} ({city['url']}) - Attempt {attempt + 1}")
            print(f"\nAccessing {city['display_name']} ({city['url']}) at {get_vietnam_time().strftime('%Y-%m-%d %H:%M:%S')}...")

            page.goto(city['url'])
            page.wait_for_selector(".aqi-value__estimated", timeout=10000)

            aqi_raw = page.query_selector(".aqi-value__estimated").text_content()
            wind_speed_raw = page.query_selector(".air-quality-forecast-container-wind__label").text_content()
            humidity_raw = page.query_selector(".air-quality-forecast-container-humidity__label").text_content()

            aqi = validate_aqi(aqi_raw)
            wind_speed = validate_wind_speed(wind_speed_raw)
            humidity = validate_humidity(humidity_raw)

            if not all([aqi, wind_speed, humidity]):
                logging.warning(f"dữ liệu lỗi {city['display_name']}: AQI={aqi_raw}, Wind={wind_speed_raw}, Humidity={humidity_raw}")
                print(f"tìm thấy dữ liệu lỗi của {city['display_name']}:")
                if not aqi: print(f"  - Lỗi AQI: {aqi_raw}")
                if not wind_speed: print(f"  - Lỗi tốc độ gió: {wind_speed_raw}")
                if not humidity: print(f"  - Lỗi độ ẩm: {humidity_raw}")
                return None

            current_time = get_vietnam_time()
            data = {
                "timestamp": current_time.isoformat(timespec='microseconds'),
                "city": city['display_name'],
                "aqi": aqi,
                "wind_speed": wind_speed,
                "humidity": humidity
            }

            logging.info(f"Cào thành công dữ liệu của {city['display_name']}: {data}")
            return data

        except Exception as e:
            logging.error(f"Lỗi dữ liệu với {city['display_name']} - Attempt {attempt + 1}: {str(e)}")
            print(f"Lỗi dữ liệu với {city['display_name']} - Attempt {attempt + 1}: {str(e)}")
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return None

def save_to_csv(data_list: List[Dict]):
    if not data_list:
        logging.warning("Không có dữ liệu để ghi CSV.")
        print("Không có dữ liệu để ghi CSV.")
        return

    columns = ["timestamp", "city", "aqi", "wind_speed", "humidity"]
    new_df = pd.DataFrame(data_list, columns=columns)

    new_df['timestamp'] = new_df['timestamp'].str.replace('T', ' ', regex=False)
    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f%z', errors='coerce')

    print("dữ liệu mới:")
    print(new_df)

    try:
        if os.path.exists(CSV_FILE):
            old_df = pd.read_csv(CSV_FILE, encoding='utf-8')
            old_df['timestamp'] = pd.to_datetime(old_df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f%z', errors='coerce')

            new_df = new_df[~new_df[['timestamp', 'city']].apply(tuple, axis=1).isin(old_df[['timestamp', 'city']].apply(tuple, axis=1))]
            if new_df.empty:
                logging.info("Không có bản ghi mới để thêm sau khi check")
                print("Không có bản ghi mới để thêm sau khi check")
                return
            combined_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        if combined_df['timestamp'].isna().any():
            logging.warning(f"Tìm thấy {combined_df['timestamp'].isna().sum()} lỗi thời gian")
            print(f"Tìm thấy  {combined_df['timestamp'].isna().sum()} dữ liệu thời gian không hợp lệ")
            combined_df = combined_df.dropna(subset=['timestamp'])

        combined_df['timestamp'] = combined_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f%z')
        combined_df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
        logging.info(f"Dữ liệu được cập nhật vào {CSV_FILE} với {len(new_df)} bản ghi moi")
        print(f"Dữ liệu được cập nhật vào {CSV_FILE} với {len(new_df)} bản ghi mới tại {get_vietnam_time().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        logging.error(f"Lỗi lưu vào CSV: {e}")
        print(f"Lỗi lưu vào CSV: {e}")

def crawl_all_cities():
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_viewport_size({"width": 1280, "height": 720})
        page.set_default_timeout(10000)

        for city in CITIES:
            print(f"\n{'=' * 50}")
            print(f"Processing {city['display_name']}...")
            data = crawl_city_data(page, city)
            if data:
                results.append(data)
            else:
                print(f"Bỏ qua dữ liệu không hợp lệ {city['display_name']}")

        browser.close()

    save_to_csv(results)
    return results

def main():
    logging.info("Bắt đầu cào dữ liệu...")
    print("Bắt đầu cào dữ liệu...")
    try:
        start_time = time.time()
        current_time = get_vietnam_time()
        print(f"\nCào dữ liệu tại thời điểm {current_time.strftime('%Y-%m-%d %H:%M:%S')}...")
        results = crawl_all_cities()
        print("Cào dữ liệu:")
        print(json.dumps(results, indent=2, ensure_ascii=False))
    except Exception as e:
        logging.error(f"Lỗi không mong muốn : {str(e)}")
        print(f"Lỗi không mong muốn: {str(e)}")
        raise

if __name__ == "__main__":
    main()
