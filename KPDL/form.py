import tkinter as tk
import joblib

def submit_form():
    # Lấy dữ liệu từ các ô nhập
    data = []
    for entry in entry_fields:
        data.append(float(entry.get()))

    model = joblib.load(r'D:\KPDL\test_joblib\decision_tree_model.joblib')
  
    # Dự đoán trên tập dữ liệu
    y_pred = model.predict([data])
    if y_pred[0] == 1:
        result_label.config(text="Xảy ra hỏa hoạn !!!", font=("Arial", 20), fg="red") 
    else:
        result_label.config(text="Không xảy ra hỏa hoạn", font=("Arial", 20), fg="blue")

def reset_form():
    # Xóa nội dung đã nhập trong các ô nhập
    for entry in entry_fields:
        entry.delete(0, tk.END)
    # Reset nhãn kết quả
    result_label.config(text="Kết quả dự đoán:", font=("Arial", 20))


# Tạo cửa sổ
root = tk.Tk()
root.title("Form Nhập Dữ Liệu")

# Tạo Frame cho phần tiêu đề
title_frame = tk.Frame(root)
title_frame.grid(row=0, column=0, columnspan=2, pady=10)

# Thêm tiêu đề cho form
form_title = tk.Label(title_frame, text="Form Test", font=("Arial", 20, "bold"), fg="red")
form_title.pack()

# Tạo Frame cho phần nhập liệu
input_frame = tk.Frame(root)
input_frame.grid(row=1, column=0, columnspan=2)

# Tạo danh sách các tên của từng ô nhập
entry_names = ["Nhiệt độ (độ C)", "Độ ẩm (%)", "Các hợp chất hữu cơ dễ bay hơi (ppb)", "Nồng độ CO2 tương đương (ppm)", "Hydro phân tử nguyên chất", "Khí ethanol nguyên chất",
               "Áp suất không khí (hPa)", "PM1.0 (Kích thước hạt vật chất < 1,0 µm)", "PM2.5 (Kích thước hạt vật chất trong [1,0 µm - 2,5 µm] )", "NC0.5 (Nồng độ số của các hạt có kích thước nhỏ hơn 0.5 µm)",
               "NC1.0 (Nồng độ số của các hạt có kích thước từ 0.5 µm đến dưới 1.0 µm)", "NC2.5 (Nồng độ số của các hạt có kích thước từ 1.0 µm đến dưới 2.5 µm)", "Bộ đếm mẫu"]

# Tạo danh sách các ô nhập
entry_fields = []
for i in range(13):
    label = tk.Label(input_frame, text=f"{entry_names[i]}:", font=("Arial", 12), fg = "blue") # Chỉnh cỡ chữ ở đây
    label.grid(row=i, column=0, padx=10, pady=5, sticky="e") # Sử dụng sticky để căn chỉnh về phía đông
    entry = tk.Entry(input_frame, font=("Arial", 12)) # Chỉnh cỡ chữ ở đây
    entry.grid(row=i, column=1, padx=10, pady=5, sticky="w") # Sử dụng sticky để căn chỉnh về phía tây
    entry_fields.append(entry)

# Tạo nút "Submit" để gửi dữ liệu
submit_button = tk.Button(root, text="Submit", command=submit_form, font=("Arial", 12)) # Chỉnh cỡ chữ ở đây
submit_button.grid(row=2, column=0, padx=10, pady=10)

# Tạo nút "Reset" để xóa dữ liệu đã nhập
reset_button = tk.Button(root, text="Reset", command=reset_form, font=("Arial", 12)) # Chỉnh cỡ chữ ở đây
reset_button.grid(row=2, column=1, padx=10, pady=10)

# Tạo Frame cho phần kết quả
result_frame = tk.Frame(root)
result_frame.grid(row=3, column=0, columnspan=2)

result_label = tk.Label(result_frame, text="Kết quả dự đoán:", font=("Arial", 20)) # Chỉnh cỡ chữ ở đây
result_label.pack()
root.geometry("800x600") # Thay đổi kích thước cửa sổ

root.mainloop()
