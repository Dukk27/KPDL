import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

# Đọc dữ liệu từ tập tin CSV
test_data = pd.read_csv(r'C:\Users\dn596\Downloads\test_smoke.csv')

# Xác định các đặc trưng và nhãn cho tập test
X_test = test_data.iloc[:, :-1]  # Loại bỏ cột nhãn để lấy các đặc trưng
y_test = test_data.iloc[:, -1]  # Nhãn thực tế

# Load mô hình đã lưu
loaded_model = joblib.load(r'D:\KPDL\test_joblib\decision_tree_model.joblib')

# Dự đoán trên tập kiểm tra
y_pred = loaded_model.predict(X_test)

# Đánh giá độ chính xác của mô hình trên tập kiểm tra
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test set (Decision Tree):", accuracy)
