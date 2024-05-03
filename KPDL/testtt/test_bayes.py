import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

# Đọc dữ liệu từ tập tin CSV
test_data = pd.read_csv(r'C:\Users\dn596\Downloads\test_smoke.csv')

# Xác định các đặc trưng và nhãn cho tập kiểm tra
X_test = test_data.iloc[:, :-1]  # Loại bỏ cột nhãn để lấy các đặc trưng
y_test_true = test_data.iloc[:, -1]   # Nhãn thực tế

# Load mô hình Naive Bayes từ file
loaded_model = joblib.load(r'D:\KPDL\test_joblib\naive_bayes_model.joblib')

# Dự đoán nhãn trên tập kiểm tra
y_pred = loaded_model.predict(X_test)

# Đánh giá độ chính xác của mô hình trên tập kiểm tra
accuracy = accuracy_score(y_test_true, y_pred)
print("Accuracy on test set (Naive Bayes):", accuracy)
