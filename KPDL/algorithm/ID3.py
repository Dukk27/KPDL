import pandas as pd
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Đọc dữ liệu từ tập tin CSV
data = pd.read_csv(r'C:\Users\dn596\Downloads\smoke_detection_training.csv')

# Xác định các đặc trưng và nhãn
X = data.iloc[:, :-1]  # Loại bỏ cột nhãn để lấy các đặc trưng
y = data.iloc[:, -1]  # Nhãn

# Phân chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

time_start = time.time()

# Xây dựng mô hình cây quyết định ID3
model = DecisionTreeClassifier(criterion='entropy') 
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Tạo ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix of ID3:")
print(cm)

# Tính độ chính xác, precision, recall và F1 score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of ID3:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)

# Tính false negative rate (FNR) và false positive rate (FPR)
tn, fp, fn, tp = cm.ravel()
fnr = fn / (fn + tp)
fpr = fp / (fp + tn)

print("False Negative Rate (FNR):", fnr)
print("False Positive Rate (FPR):", fpr)

# Lưu mô hình đã huấn luyện
joblib.dump(model, r'D:\KPDL\test_joblib\decision_tree_model.joblib')

time_end = time.time()
print("time =", time_end - time_start)

# # In cây quyết định
# tree_rules = export_text(model, feature_names=list(X.columns))
# print(tree_rules)