import pandas as pd

# Đọc dữ liệu từ file csv mà không có hàng tiêu đề
df = pd.read_csv(r'C:\Users\dn596\Downloads\smoke_detection_iot.csv')

original_list = []


seen_rows = []

# Lặp qua từng hàng trong DataFrame và lưu vào danh sách
for index, row in df.iterrows():
    row_data = row.tolist() 
    original_list.append(row_data) # Chuyển đổi hàng thành mảng
    # Kiểm tra xem hàng đã xuất hiện trước đó hay chưa
    if row_data not in seen_rows:
        seen_rows.append(row_data)

print("Original_list length: " +  str(len(original_list)))
print("Cleaned: " + str(len(seen_rows)))

cleaned_file_path = "dataset-cleaned.txt"

# # Mở file để ghi dữ liệu
# with open(cleaned_file_path, "w") as file:
#     # Viết từng hàng từ original_list vào file
#     for row in original_list:
#         file.write(','.join(map(str, row)) + '\n')

# print(f"Dữ liệu đã được xuất ra file {cleaned_file_path}.")