# Import các thư viện cần thiết
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Đọc dữ liệu từ file CSV
df = pd.read_csv("data/comb_PM25_wind_Hanoi_2018_v3.csv")

# Kiểm tra và thay thế các giá trị NaN bằng 0 trong toàn bộ DataFrame
df.fillna(0, inplace=True)

# Xác định biến đầu vào X và biến mục tiêu y
X = df[
    [
        "T2MDEW",
        "T2M",
        "PS",
        "TQV",
        "TQL",
        "H1000",
        "DISPH",
        "FRCAN",
        "HLML",
        "RHOA",
        "CIG",
        "WS",
        "CLDCR",
        "v_2m",
        "v_50m",
        "v_850",
    ]
]
y = df["PM2.5"]

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Khởi tạo mô hình Linear Regression
model = LinearRegression()

# Huấn luyện mô hình trên tập huấn luyện
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Tính toán lỗi bình phương trung bình (MSE) trên tập kiểm tra
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse}")

# Tính toán lỗi tuyệt đối trung bình (MAE) trên tập kiểm tra
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error on Test Set: {mae}")

# In ra hệ số của mô hình
coefficients = pd.DataFrame({"Tham số": X.columns, "Hệ số": model.coef_})
print(coefficients)

# In ra hệ số chặn (intercept)
print(f"Hệ số chặn (Intercept): {model.intercept_}")

# Tính R-squared trên tập kiểm tra
r_squared = model.score(X_test, y_test)
print(f"R-squared on Test Set: {r_squared}")

