import cv2
import numpy as np
import pandas as pd
import os
from deepface import DeepFace
from PIL import Image
from pymongo import MongoClient

# Kết nối đến MongoDB
client = MongoClient('mongodb+srv://congdinh2412:SX3U5c8VLUTHZQfV@cluster0.3i7aqal.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['image_features']
collection = db['gender_features']

def convert_gender(gender):
    """
    Chuyển đổi giới tính từ tiếng Anh sang tiếng Việt
    """
    return "Nam" if gender == "Man" else "Nữ" if gender == "Woman" else gender

def extract_gender(image_path):
    """
    Trích xuất giới tính từ ảnh sử dụng DeepFace
    """
    try:
        # Đọc ảnh bằng PIL và chuyển sang định dạng phù hợp
        pil_image = Image.open(image_path).convert('RGB')
        # Chuyển đổi sang numpy array
        image = np.array(pil_image)
        
        # Phân tích khuôn mặt và giới tính bằng DeepFace
        result = DeepFace.analyze(
            img_path=image,
            actions=['gender'],
            enforce_detection=False,
            detector_backend='opencv',
            silent=True  # Tắt các thông báo không cần thiết
        )
        
        # Lấy kết quả giới tính
        if isinstance(result, list):
            result = result[0]
        
        # Chuyển đổi sang tiếng Việt
        gender = convert_gender(result['dominant_gender'])
        return gender
            
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {os.path.basename(image_path)}: {str(e)}")
        return None

def process_images(image_dir):
    """
    Xử lý tất cả ảnh trong thư mục và lưu kết quả vào MongoDB
    """
    try:
        print(f"\nBắt đầu xử lý thư mục: {image_dir}")
        
        # Kiểm tra thư mục có tồn tại không
        if not os.path.exists(image_dir):
            print(f"Thư mục {image_dir} không tồn tại!")
            return
        
        # Lấy danh sách các file ảnh
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"Không tìm thấy ảnh trong thư mục {image_dir}")
            return
            
        print(f"Tìm thấy {len(image_files)} ảnh")
        
        # Tạo danh sách để lưu kết quả
        results = []
        
        # Xử lý từng ảnh
        for i, image_file in enumerate(image_files, 1):
            print(f"Đang xử lý ảnh {i}/{len(image_files)}: {image_file}")
            image_path = os.path.join(image_dir, image_file)
            
            # Trích xuất giới tính
            gender = extract_gender(image_path)
            
            if gender:
                results.append({
                    'filename': image_file,
                    'gender': gender
                })
                print(f"Giới tính: {gender}")
        
        # Lưu kết quả vào MongoDB
        if results:
            collection.insert_many(results)
            print(f"\nĐã xử lý thành công {len(results)}/{len(image_files)} ảnh")
            print(f"Đã lưu kết quả vào MongoDB collection 'gender_features'")
        else:
            print("\nKhông có ảnh nào được xử lý thành công")
            
    except Exception as e:
        print(f"Lỗi khi xử lý thư mục {image_dir}: {str(e)}")

def process_all_directories():
    """
    Xử lý tất cả các thư mục ảnh
    """
    # Đường dẫn đến thư mục chứa ảnh
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), 'Data')
    
    # Kiểm tra xem thư mục tồn tại không
    if not os.path.exists(data_dir):
        print(f"Thư mục ảnh không tồn tại: {data_dir}")
        return
    
    # Xóa collection cũ nếu tồn tại
    collection.drop()
    
    print("Bắt đầu trích xuất giới tính từ thư mục...")
    
    # Xử lý thư mục Data
    process_images(data_dir)
    
    # In ra một vài ví dụ
    print("\nVí dụ về dữ liệu đã lưu:")
    for doc in collection.find().limit(3):
        print(doc)
        
    print(f"\nTổng số bản ghi trong collection: {collection.count_documents({})}")
    print("\nĐã hoàn thành xử lý tất cả các thư mục!")

if __name__ == "__main__":
    process_all_directories()