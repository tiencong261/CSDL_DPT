# Ứng dụng Tìm kiếm Ảnh Tương tự

Ứng dụng web Flask để tìm kiếm ảnh tương tự dựa trên đặc trưng ảnh và MongoDB.

## Cài đặt

1. Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

2. Đảm bảo bạn đã có kết nối đến MongoDB Atlas (thông tin kết nối đã được cấu hình trong code).

## Cấu trúc thư mục

```
code_ok/
├── app.py                  # Ứng dụng Flask chính
├── find_similar_images.py  # Module tìm kiếm ảnh tương tự
├── tien_xu_ly.py           # Module tiền xử lý ảnh
├── static/
│   ├── css/                # CSS files
│   ├── js/                 # JavaScript files
│   └── uploads/            # Thư mục lưu ảnh tải lên
└── templates/
    ├── index.html          # Trang chủ
    └── result.html         # Trang kết quả tìm kiếm
```

## Chạy ứng dụng

Để chạy ứng dụng, thực hiện các lệnh sau:

```bash
cd code_ok
python app.py
```

Sau đó, mở trình duyệt web và truy cập địa chỉ: `http://localhost:5000`

## Sử dụng

1. Tải lên ảnh từ máy tính hoặc nhập URL ảnh từ internet.
2. Hệ thống sẽ tự động xử lý và tìm kiếm các ảnh tương tự.
3. Kết quả hiển thị bao gồm ảnh đầu vào (đã qua tiền xử lý) và 3 ảnh tương tự nhất.

## Lưu ý

- Ứng dụng sử dụng MongoDB Atlas để lưu trữ đặc trưng ảnh.
- Đảm bảo rằng thư mục `Data` chứa các ảnh mẫu đã được đặt đúng vị trí (cùng cấp với thư mục `code_ok`).
- Nếu gặp vấn đề về kết nối MongoDB, hãy kiểm tra thông tin kết nối trong file `find_similar_images.py`.
