import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from skimage.feature import hog
from pymongo import MongoClient

# Kết nối đến MongoDB
client = MongoClient('mongodb+srv://congdinh2412:SX3U5c8VLUTHZQfV@cluster0.3i7aqal.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['image_features']
collection = db['hog_features']

def extract_hog_features(image):
    """
    Trích xuất đặc trưng hình dạng sử dụng HOG
    """
    # Chuyển ảnh sang grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Trích xuất HOG features
    features = hog(gray, 
                  orientations=9,
                  pixels_per_cell=(8, 8),
                  cells_per_block=(8, 8),
                  block_norm='L2-Hys')
    return features

def reduce_hog_statistical(hog_vector, n_parts=10):
    hog_vector = np.array(hog_vector)
    splits = np.array_split(hog_vector, n_parts)
    reduced = np.array([part.mean() for part in splits])
    return reduced

def process_images():
    # Đường dẫn đến thư mục chứa ảnh
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(os.path.dirname(current_dir), 'Data')
    
    # Kiểm tra xem thư mục tồn tại không
    if not os.path.exists(image_dir):
        print(f"Thư mục ảnh không tồn tại: {image_dir}")
        return
    
    # Danh sách để lưu kết quả
    results = []
    
    # Lấy danh sách các file ảnh
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Xóa collection cũ nếu tồn tại
    collection.drop()
    
    print("Đang trích xuất đặc trưng HOG...")
    print(f"Tìm thấy {len(image_files)} ảnh trong thư mục {image_dir}")
    
    # Xử lý từng ảnh
    for filename in tqdm(image_files):
        try:
            # Đọc ảnh
            image_path = os.path.join(image_dir, filename)
            
            # Đọc ảnh bằng numpy trước
            image = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            
            if image is None:
                print(f"Không thể đọc ảnh: {filename}")
                continue
                
            # Trích xuất đặc trưng HOG
            features = extract_hog_features(image)
            reduced_features = reduce_hog_statistical(features, n_parts=10)
            
            # Tạo dictionary chứa tên file và các đặc trưng đã giảm chiều
            result = {'filename': filename}
            for i, feature in enumerate(reduced_features):
                result[f'reduced_mean_{i}'] = float(feature)
            results.append(result)
            
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {filename}: {str(e)}")
    
    # Lưu vào MongoDB
    if results:
        collection.insert_many(results)
        print(f"\nĐã lưu {len(results)} bản ghi vào MongoDB collection 'hog_features'")
        
        # In ra một vài ví dụ (chỉ in ra 5 đặc trưng đầu tiên)
        print("\nVí dụ về dữ liệu đã lưu (chỉ hiển thị 5 đặc trưng đầu tiên):")
        for doc in collection.find().limit(1):
            sample_doc = {k: doc[k] for k in list(doc.keys())[:6]}  # _id + filename + 4 features
            print(sample_doc)
    else:
        print("Không có dữ liệu để lưu")

if __name__ == "__main__":
    process_images() 