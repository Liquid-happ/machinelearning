#include "WiFi.h"
#include <WiFiMulti.h> // Thư viện WiFiMulti
#include <Wire.h>
#include "ThingSpeak.h"
#include <Adafruit_Sensor.h>
#include <Adafruit_BME680.h>
#include <Adafruit_PM25AQI.h>
#include "bsec.h"
#include <Adafruit_SSD1306.h>
#include <TinyGPSPlus.h> // Thư viện cho GPS

// WiFiMulti
WiFiMulti wifiMulti;

// Danh sách các mạng WiFi
const char* wifiNetworks[][2] = {
  {"Vo tuyen 217", "votuyen217@"},
  {"Tang1", "999999999"},
  {"Liquid", "88888888"},
  // Thêm các mạng khác nếu cần, ví, dụ: {"SSID", "Password"}
};

// ThingSpeak - Kênh 1 (Cảm biến)
#define CHANNEL_ID_SENSOR 2897503
#define API_KEY_SENSOR "IZ02DZW7RWBUQ94B"

// ThingSpeak - Kênh 2 (GPS)
#define CHANNEL_ID_GPS 2960009
#define API_KEY_GPS "MNU8BPCWBAXSAGVS"

WiFiClient client; // Client chung cho cả hai kênh

// Sensors
Bsec iaqSensor;
Adafruit_PM25AQI aqiSensor;
Adafruit_BME680 bme;
TinyGPSPlus gps; // Đối tượng GPS

// PMS5003 Serial pins
#define PMS_RX_PIN 16
#define PMS_TX_PIN 17

// GPS Serial pins
#define GPS_RX_PIN 26
#define GPS_TX_PIN 27

// OLED
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
TwoWire I2C_2 = TwoWire(1);
Adafruit_SSD1306 display2(SCREEN_WIDTH, SCREEN_HEIGHT, &I2C_2);
#define SDA_2 32
#define SCL_2 33

// Timing
unsigned long lastTime = 0;
const long interval = 15000; // 15 giây để tuân thủ giới hạn ThingSpeak miễn phí

// LED pins
void setup() {
  Serial.begin(115200);
  Serial.println("Bắt đầu khởi động...");

  // Thêm các mạng WiFi vào WiFiMulti
  for (int i = 0; i < sizeof(wifiNetworks) / sizeof(wifiNetworks[0]); i++) {
    wifiMulti.addAP(wifiNetworks[i][0], wifiNetworks[i][1]);
    Serial.printf("Đã thêm WiFi: %s\n", wifiNetworks[i][0]);
  }

  // Kết nối WiFi
  Serial.print("Đang kết nối WiFi...");
  while (wifiMulti.run() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }
  Serial.println("\nWiFi đã kết nối!");
  Serial.print("Địa chỉ IP: ");
  Serial.println(WiFi.localIP());
  Serial.print("Đã kết nối với SSID: ");
  Serial.println(WiFi.SSID());
  ThingSpeak.begin(client);

  // Khởi tạo BME680
  if (!bme.begin()) {
    Serial.println("Không tìm thấy cảm biến BME680!");
    while (1);
  }

  // Khởi tạo BSEC
  iaqSensor.begin(0x77, Wire);
  bsec_virtual_sensor_t sensorList[] = {
    BSEC_OUTPUT_IAQ, BSEC_OUTPUT_RAW_TEMPERATURE, BSEC_OUTPUT_RAW_HUMIDITY,
    BSEC_OUTPUT_RAW_PRESSURE, BSEC_OUTPUT_RAW_GAS
  };
  iaqSensor.updateSubscription(sensorList, 5, BSEC_SAMPLE_RATE_LP);
  Serial.println("Cảm biến BME680 đã được cấu hình với BSEC!");

  // Khởi tạo PMS5003
  Serial1.begin(9600, SERIAL_8N1, PMS_RX_PIN, PMS_TX_PIN);
  if (!aqiSensor.begin_UART(&Serial1)) {
    Serial.println("Không tìm thấy PMS5003, tiếp tục chạy không có nó.");
  } else {
    Serial.println("PMS5003 đã kết nối thành công!");
  }

  // Khởi tạo GPS NEO-6M
  Serial2.begin(9600, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
  Serial.println("Đã khởi tạo GPS NEO-6M!");

  // Khởi tạo OLED
  I2C_2.begin(SDA_2, SCL_2);
  if (!display2.begin(SSD1306_PAGEADDR, 0x3C)) {
    Serial.println(F("OLED SSD1306 không khởi tạo được!"));
    while (true);
  }
  display2.clearDisplay();
  display2.display();

  // Cấu hình chân đầu ra
  pinMode(15, OUTPUT);
  pinMode(4, OUTPUT);
  pinMode(2, OUTPUT);
}

// Hàm tính AQI từ PM2.5
int calculatePM25AQI(float pm25) {
  struct AQIRange {
    float cLow, cHigh;
    int iLow, iHigh;
  };

  AQIRange ranges[] = {
    {0.0, 12.0, 0, 50},      // Good
    {12.0, 35.4, 50, 100},   // Moderate
    {35.4, 55.4, 100, 150},  // Unhealthy for Sensitive Groups
    {55.4, 150.4, 150, 200}, // Unhealthy
    {150.4, 250.4, 200, 300}, // Very Unhealthy
    {250.4, 500.0, 300, 500} // Hazardous
  };

  for (auto& r : ranges) {
    if (pm25 >= r.cLow && pm25 <= r.cHigh) {
      return (int)((r.iHigh - r.iLow) * (pm25 - r.cLow) / (r.cHigh - r.cLow) + r.iLow);
    }
  }
  return -1;
}

void loop() {
  // Cập nhật dữ liệu GPS
  while (Serial2.available() > 0) {
    gps.encode(Serial2.read());
  }

  // Kiểm tra và duy trì kết nối WiFi
  if (wifiMulti.run() != WL_CONNECTED) {
    Serial.println("WiFi bị mất kết nối! Đang thử kết nối lại...");
    while (wifiMulti.run() != WL_CONNECTED) {
      Serial.print(".");
      delay(500);
    }
    Serial.println("\nWiFi đã kết nối lại!");
    Serial.print("Địa chỉ IP: ");
    Serial.println(WiFi.localIP());
    Serial.print("Đã kết nối với SSID: ");
    Serial.println(WiFi.SSID());
  }

  if (millis() - lastTime >= interval) {
    lastTime = millis();

    if (iaqSensor.run()) {
      float temperature = bme.readTemperature();
      float humidity = bme.readHumidity();
      float pressure = bme.readPressure() / 100.0;
      float iaq = iaqSensor.iaq;
      float gasResistance = iaqSensor.gasResistance;

      Serial.println("=== Dữ liệu từ BME680 (BSEC) ===");
      Serial.printf("Nhiệt độ: %.2f °C\n", temperature);
      Serial.printf("Độ ẩm: %.2f %%\n", humidity);
      Serial.printf("Áp suất: %.2f hPa\n", pressure);
      Serial.printf("Chỉ số IAQ: %.2f\n", iaq);
      Serial.printf("Gas Resistance: %.2f KΩ\n", gasResistance);

      PM25_AQI_Data data;
      bool pmsAvailable = aqiSensor.read(&data);

      int pm25AQI = -1;
      Serial.println("=== Dữ liệu từ PMS5003 ===");
      if (pmsAvailable) {
        Serial.printf("PM1.0: %d µg/m³\n", data.pm10_standard);
        Serial.printf("PM2.5: %d µg/m³\n", data.pm25_standard);
        Serial.printf("PM10: %d µg/m³\n", data.pm100_standard);

        pm25AQI = calculatePM25AQI((float)data.pm25_standard);
        Serial.printf("Chỉ số AQI từ PM2.5: %d\n", pm25AQI);
      } else {
        Serial.println("Không thể đọc dữ liệu từ PMS5003!");
      }

      // Dữ liệu GPS
      float latitude = 0.0;
      float longitude = 0.0;
      int satellites = 0;
      float speed = 0.0;
      float altitude = 0.0;
      bool gpsAvailable = false;
      if (gps.location.isValid() && gps.satellites.isValid() && gps.speed.isValid() && gps.altitude.isValid()) {
        latitude = gps.location.lat();
        longitude = gps.location.lng();
        satellites = gps.satellites.value();
        speed = gps.speed.kmph();
        altitude = gps.altitude.meters();
        gpsAvailable = true;
        Serial.println("=== Dữ liệu từ GPS NEO-6M ===");
        Serial.printf("Vĩ độ: %.6f\n", latitude);
        Serial.printf("Kinh độ: %.6f\n", longitude);
        Serial.printf("Số vệ tinh: %d\n", satellites);
        Serial.printf("Tốc độ: %.2f km/h\n", speed);
        Serial.printf("Độ cao: %.2f m\n", altitude);
      } else {
        Serial.println("Không nhận được tín hiệu GPS đầy đủ!");
      }

      // Gửi dữ liệu lên ThingSpeak - Kênh 1 (Cảm biến)
      if (WiFi.status() == WL_CONNECTED) {
        ThingSpeak.setField(1, temperature);
        ThingSpeak.setField(2, humidity);
        ThingSpeak.setField(3, pressure);
        ThingSpeak.setField(4, gasResistance);
        ThingSpeak.setField(5, iaq);
        if (pmsAvailable) {
          ThingSpeak.setField(6, data.pm10_standard);
          ThingSpeak.setField(7, data.pm25_standard);
          ThingSpeak.setField(8, data.pm100_standard);
        }

        int status = ThingSpeak.writeFields(CHANNEL_ID_SENSOR, API_KEY_SENSOR);
        if (status == 200) {
          Serial.println("Dữ liệu cảm biến gửi thành công lên ThingSpeak (Kênh 1)!");
        } else {
          Serial.printf("Lỗi gửi dữ liệu cảm biến! Mã lỗi: %d\n", status);
        }

        // Gửi dữ liệu GPS lên ThingSpeak - Kênh 2
        if (gpsAvailable) {
          ThingSpeak.setField(1, latitude);
          ThingSpeak.setField(2, longitude);
          ThingSpeak.setField(3, satellites);
          ThingSpeak.setField(4, speed);
          ThingSpeak.setField(5, altitude);

          status = ThingSpeak.writeFields(CHANNEL_ID_GPS, API_KEY_GPS);
          if (status == 200) {
            Serial.println("Dữ liệu GPS gửi thành công lên ThingSpeak (Kênh 2)!");
          } else {
            Serial.printf("Lỗi gửi dữ liệu GPS! Mã lỗi: %d\n", status);
          }
        } else {
          Serial.println("Không gửi dữ liệu GPS do thiếu tín hiệu!");
        }
      } else {
        Serial.println("WiFi bị mất kết nối!");
      }

      // LED cảnh báo
      if (iaq <= 100 || data.pm25_standard <= 35.5) {
        digitalWrite(4, HIGH);
        digitalWrite(15, LOW);
        digitalWrite(2, LOW);
      } else if ((iaq > 100 && iaq <= 200) || (data.pm25_standard > 35.5 && data.pm25_standard < 125.5)) {
        digitalWrite(15, HIGH);
        digitalWrite(4, LOW);
        digitalWrite(2, LOW);
      } else {
        digitalWrite(15, LOW);
        digitalWrite(4, LOW);
        digitalWrite(2, HIGH);
      }

      // Hiển thị trên OLED (giữ nguyên như mã gốc)
      display2.clearDisplay();

      // Nhóm IAQ + PM2.5 AQI
      display2.setTextSize(2);
      display2.setTextColor(SSD1306_WHITE);
      display2.setCursor(0, 0);
      display2.print("IAQ:");
      display2.setCursor(48, 0);
      display2.print(iaq, 0);

      if (pm25AQI != -1) {
        display2.setCursor(0, 18);
        display2.print("AQI:");
        display2.setCursor(48, 18);
        display2.print(pm25AQI);
      }

      // Nhóm PM2.5 + gas
      display2.setTextSize(1);
      display2.setCursor(0, 40);
      display2.print("PM2.5: ");
      display2.print(data.pm25_standard);
      display2.print(" ug/m3");

      display2.setCursor(0, 52);
      display2.print("Gas_res: ");
      display2.print(gasResistance, 0);
      display2.print(" KOhm");

      display2.display();
    } else {
      Serial.println("Lỗi cập nhật dữ liệu từ BME680!");
    }
  }
}
