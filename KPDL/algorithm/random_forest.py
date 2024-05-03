import pandas as pd
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# Load dữ liệu từ file CSV
data = pd.read_csv(r'C:\Users\dn596\Downloads\smoke_detection_training.csv')

# Chia dữ liệu thành features (X) và target (y)
X = data.drop(columns=["Fire Alarm"])  # Thay "target_column" bằng tên cột chứa biến mục tiêu
y = data["Fire Alarm"]

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

time_start = time.time()
# Tạo mô hình RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Huấn luyện mô hình trên tập huấn luyện
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Tạo ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix of Random Forest:")
print(cm)

# Tính precision, recall và F1 score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Tính độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of RandomForest:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)

# Tính false negative rate (FNR) và false positive rate (FPR)
tn, fp, fn, tp = cm.ravel()
fnr = fn / (fn + tp)
fpr = fp / (fp + tn)

print("False Negative Rate (FNR):", fnr)
print("False Positive Rate (FPR):", fpr)

# Lưu mô hình vào file
joblib.dump(model, r'D:\KPDL\test_joblib\random_forest_model.joblib')

time_end = time.time()
print("time =", time_end - time_start)