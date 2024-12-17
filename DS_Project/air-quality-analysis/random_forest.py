# Import các thư viện cần thiết
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

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
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, y_train)
forest_ytrain_p = forest_reg.predict(X_train)

forest_reg.score(X_test, y_test)
mse_train = mean_squared_error(y_train, forest_ytrain_p)
rmse_train = np.sqrt(mse_train)
print(mse_train)
